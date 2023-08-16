{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "47e5ab57",
   "metadata": {},
   "source": [
    "# IRIS FLOWER CLASSIFICATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "07fb38a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import the libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import random as rnd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "b0fde4ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "data= pd.read_csv(\"C:\\\\Users\\\\visha\\\\OneDrive\\\\Documents\\\\csv\\\\iris flower classification task-2.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "aaf5dde9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal_length  sepal_width  petal_length  petal_width         species\n",
       "0             5.1          3.5           1.4          0.2     Iris-setosa\n",
       "1             4.9          3.0           1.4          0.2     Iris-setosa\n",
       "2             4.7          3.2           1.3          0.2     Iris-setosa\n",
       "3             4.6          3.1           1.5          0.2     Iris-setosa\n",
       "4             5.0          3.6           1.4          0.2     Iris-setosa\n",
       "..            ...          ...           ...          ...             ...\n",
       "145           6.7          3.0           5.2          2.3  Iris-virginica\n",
       "146           6.3          2.5           5.0          1.9  Iris-virginica\n",
       "147           6.5          3.0           5.2          2.0  Iris-virginica\n",
       "148           6.2          3.4           5.4          2.3  Iris-virginica\n",
       "149           5.9          3.0           5.1          1.8  Iris-virginica\n",
       "\n",
       "[150 rows x 5 columns]"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "1e6b6cfe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal_length  sepal_width  petal_length  petal_width      species\n",
       "0           5.1          3.5           1.4          0.2  Iris-setosa\n",
       "1           4.9          3.0           1.4          0.2  Iris-setosa\n",
       "2           4.7          3.2           1.3          0.2  Iris-setosa\n",
       "3           4.6          3.1           1.5          0.2  Iris-setosa\n",
       "4           5.0          3.6           1.4          0.2  Iris-setosa"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "2562b316",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.054000</td>\n",
       "      <td>3.758667</td>\n",
       "      <td>1.198667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.433594</td>\n",
       "      <td>1.764420</td>\n",
       "      <td>0.763161</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       sepal_length  sepal_width  petal_length  petal_width\n",
       "count    150.000000   150.000000    150.000000   150.000000\n",
       "mean       5.843333     3.054000      3.758667     1.198667\n",
       "std        0.828066     0.433594      1.764420     0.763161\n",
       "min        4.300000     2.000000      1.000000     0.100000\n",
       "25%        5.100000     2.800000      1.600000     0.300000\n",
       "50%        5.800000     3.000000      4.350000     1.300000\n",
       "75%        6.400000     3.300000      5.100000     1.800000\n",
       "max        7.900000     4.400000      6.900000     2.500000"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "d2216947",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 5 columns):\n",
      " #   Column        Non-Null Count  Dtype  \n",
      "---  ------        --------------  -----  \n",
      " 0   sepal_length  150 non-null    float64\n",
      " 1   sepal_width   150 non-null    float64\n",
      " 2   petal_length  150 non-null    float64\n",
      " 3   petal_width   150 non-null    float64\n",
      " 4   species       150 non-null    object \n",
      "dtypes: float64(4), object(1)\n",
      "memory usage: 6.0+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "25782b03",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal_length  sepal_width  petal_length  petal_width  species\n",
       "0           False        False         False        False    False\n",
       "1           False        False         False        False    False\n",
       "2           False        False         False        False    False\n",
       "3           False        False         False        False    False\n",
       "4           False        False         False        False    False\n",
       "..            ...          ...           ...          ...      ...\n",
       "145         False        False         False        False    False\n",
       "146         False        False         False        False    False\n",
       "147         False        False         False        False    False\n",
       "148         False        False         False        False    False\n",
       "149         False        False         False        False    False\n",
       "\n",
       "[150 rows x 5 columns]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "d9d7589f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sepal_length    0\n",
       "sepal_width     0\n",
       "petal_length    0\n",
       "petal_width     0\n",
       "species         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# preprocessing the datar\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "b628b023",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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p1WrVunAgQN66KGHErE7AACQZBISQO68804dPXpUTz75pA4dOqSRI0fq9ddfV3FxcSJ2168EAgE98cQTHd5WSgfp2nu69i3Rezr2nq59S+nbe6L69jkX7f/LAAAAeINrwQAAAHMEEAAAYI4AAgAAzBFAAACAOQJILyxbtkw+n0/z5s3rckxNTY18Pl+H20cffWRXqAdCoVCHHgoKCrp9zObNmzV69GgNGDBAw4cP13PPPWdUrXdi7TtV5rvVZ599pnvuuUeDBw9Wdna2vv3tb6u2trbbx6TCvEux954Kcz9s2LBOeygvL+/yMaky37H2ngrzLUmnT5/WL37xC5WUlGjgwIEaPny4nnzySbW0tHT7OC/mPWFfxZ7qduzYoVWrVulb3/pWVOP37t3b7hvkLr744kSVljBXXnml3nrrrbblzMzMLsfW19frlltu0QMPPKC1a9fq3Xff1cMPP6yLL75Yt99+u0W5noml71apMN/Hjh3T+PHjNXnyZL3xxhvKy8vTv//9726/qThV5j2e3lsl89zv2LFDZ86caVvevXu3SktLdccdd3Q6PlXmW4q991bJPN+S9PTTT+u5557Tiy++qCuvvFI7d+7UD3/4Q1144YWaO3dup4/xbN57ffW5NHT8+HE3YsQIV11d7SZOnOjmzp3b5di3337bSXLHjh0zqy8RnnjiCXf11VdHPX7+/Pnu8ssvb7fuwQcfdOPGjfO4ssSKte9UmW/nnHvsscfchAkTYnpMqsx7PL2n0ty3mjt3rrv00ktdS0tLp/enynx3pqfeU2W+p06d6u67775262bMmOHuueeeLh/j1bzzFkwcysvLNXXqVH33u9+N+jGjRo1SMBjUjTfeqLfffjuB1SVOXV2dCgsLVVJSorvuukv79u3rcux7772nsrKydutuuukm7dy5U5FIJNGleiqWvlulwny/9tprGjNmjO644w7l5eVp1KhRev7557t9TKrMezy9t0qFuZekU6dOae3atbrvvvu6vDBoqsz3uaLpvVWyz/eECRP017/+VR9//LEk6R//+Ie2bt2qW265pcvHeDXvBJAYrVu3TrW1tVq2bFlU44PBoFatWqWXX35Zr7zyir7xjW/oxhtv1JYtWxJcqbfGjh2rNWvWqKqqSs8//7waGhp0/fXX6+jRo52Ob2ho6HDxwfz8fJ0+fVpHjhyxKNkTsfadKvMtSfv27dPKlSs1YsQIVVVV6aGHHtJPfvITrVmzpsvHpMq8x9N7Ks29JL366qv68ssvNXv27C7HpMp8nyua3lNlvh977DHdfffduvzyy+X3+zVq1CjNmzdPd999d5eP8WzeYzpfkuYOHDjg8vLy3K5du9rW9fQWTGduvfVWN23aNI+rs3XixAmXn5/vfv3rX3d6/4gRI9zSpUvbrdu6dauT5A4dOmRRYkL01HdnknW+/X6/u+6669qtmzNnTrenWVNl3uPpvTPJOvfOOVdWVuZuvfXWbsekynyfK5reO5OM8/3SSy+5oUOHupdeesn985//dGvWrHGDBg1yL7zwQpeP8WreOQMSg9raWh0+fFijR49WVlaWsrKytHnzZv32t79VVlZWuw8wdWfcuHGqq6tLcLWJdf755+uqq67qso+CggI1NDS0W3f48GFlZWVp8ODBFiUmRE99dyZZ5zsYDOqb3/xmu3VXXHGFDhw40OVjUmXe4+m9M8k6959++qneeust3X///d2OS5X5Plu0vXcmGef7Zz/7mRYsWKC77rpLV111le6991498sgj3Z7l92reCSAxuPHGG/XBBx9o165dbbcxY8boBz/4gXbt2hXVf0dI0vvvv69gMJjgahMrHA7rww8/7LKP6667TtXV1e3Wbdq0SWPGjJHf77coMSF66rszyTrf48eP1969e9ut+/jjj7u9qGSqzHs8vXcmWed+9erVysvL09SpU7sdlyrzfbZoe+9MMs53c3OzMjLaR4HMzMxu/w3Xs3mP+7wNnHMd34JZsGCBu/fee9uWf/Ob37gNGza4jz/+2O3evdstWLDASXIvv/xyH1Qbv5/+9KeupqbG7du3z23fvt3deuutLicnx+3fv98517Hvffv2uezsbPfII4+4f/3rX+73v/+98/v97o9//GNftRCXWPtOlfl2zrm///3vLisryy1ZssTV1dW5P/zhDy47O9utXbu2bUyqzns8vafK3J85c8Zdcskl7rHHHutwX6rOd6tYek+V+Z41a5b7+te/7v7yl7+4+vp698orr7ghQ4a4+fPnt41J1LwTQHrp3AAya9YsN3HixLblp59+2l166aVuwIAB7mtf+5qbMGGC27hxo32hvXTnnXe6YDDo/H6/KywsdDNmzHB79uxpu//cvp1zrqamxo0aNcqdd955btiwYW7lypXGVfderH2nyny3+vOf/+xGjhzpAoGAu/zyy92qVava3Z+q8+5c7L2nytxXVVU5SW7v3r0d7kvl+XYutt5TZb6bmprc3Llz3SWXXOIGDBjghg8f7hYuXOjC4XDbmETNu88556I/XwIAANB7fAYEAACYI4AAAABzBBAAAGCOAAIAAMwRQAAAgDkCCAAAMEcAAQAA5gggAADAHAEEAACYI4AAAABzBBAAAGCOAAIAAMz9P3KR6gRWZJryAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# exploratory data analysis\n",
    "data['sepal_length'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "57af00f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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qLi4OOoY1rr9htL3Pqz1+C1dulSPP/64vvOd71x3LOvdPrH0vU+86z1hny7sJMuyVFFRoW9/+9uaMmXKoONaWloiPlgwOztb3d3dOn/+vHw+X6JLTSpD7bvP59Mrr7yiadOmKRgM6o9//KMeeeQR1dXV6aGHHhrBim9sx44dU3Fxsa5cuaJx48Zp165duvPOO6OOZa3bJ5a+s9btUVNTo8bGRjU0NAxpPOvdHrH23a71npTB5JlnntH777+vd99997pjXS7XgG3r/94I98v7cX1D7fvkyZM1efLk8HZxcbHOnDmjX/3qV5ysYzB58mQdPXpUly5d0s6dO7Vo0SLV19cP+kuStW6PWPrOWo/fmTNn9Oyzz6q2tlY33XTTkG/Heo/PcPpu13pPuqdyli1bprfeektvv/22Jk2adM2xOTk5amlpGbDv3LlzSktL080335zIMpNOLH2PZsaMGTp58mQCKkteY8aM0W233aaioiJt2LBBd999t1566aWoY1nr9oml79Gw1mPT2Nioc+fOadq0aUpLS1NaWprq6+v18ssvKy0tTT09PRG3Yb3Hbzh9j2Y46z1prphYlqVly5Zp165dqqurU0FBwXVvU1xcrL/97W8D9tXW1qqoqEhutztRpSaV4fQ9miNHjnB5NU6WZSkYDEY9xlpPnGv1PRrWemweeeQRHTt2bMC+H/zgB7rjjjv0wgsvKDU1NeI2rPf4Dafv0Qxrvcf10lmD/OQnP7HGjx9v1dXVWc3NzeGvzs7O8JiVK1daCxcuDG//97//tdLT063nnnvO+te//mW99tprltvttv785z878RBuSMPp+69//Wtr165d1n/+8x/rgw8+sFauXGlJsnbu3OnEQ7ghrVq1yjpw4IDV1NRkvf/++9bq1autlJQUq7a21rIs1nqixNp31npifPmvQ1jvI+N6fbdrvSfNFZOqqipJUklJyYD927Zt0+LFiyVJzc3NOn36dPhYQUGB9uzZo+eee06/+c1vlJubq5dffllPPvnkSJV9wxtO369evarnn39en3zyicaOHau77rpLu3fv1pw5c0aq7Bve2bNntXDhQjU3N2v8+PGaOnWq9u3bp9LSUkms9USJte+s9ZHBendGota7y7L+7xVBAAAADku6F78CAIAbF8EEAAAYg2ACAACMQTABAADGIJgAAABjEEwAAIAxCCYAAMAYBBMAAGAMggkAADAGwQQAABiDYAIAAIxBMAEAAMb4/28eBoGLw1LgAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['sepal_width'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "ff4d34fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['petal_length'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "f30aab30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAh8AAAGdCAYAAACyzRGfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAk9ElEQVR4nO3df2xVd/3H8dcFLgc6epmstr0Nteu029TKMmGDsh/AZrt1C5ExjbFmgUQdcwzFZmH8CPFWXZnkG2QJDp0uyGIaiCJzCRtwjWvZUjHQQGTokMXCcGvXwKC3tOxyac/3D+0Nl3thPbfnfs697fOR3JBz7ud+zrvv+7m3L869vddn27YtAAAAQ8Z4XQAAABhdCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjBrndQFXGhgY0AcffKD8/Hz5fD6vywEAAENg27Z6enpUUlKiMWOufW4j68LHBx98oNLSUq/LAAAAaTh16pSmTp16zTFZFz7y8/Ml/bf4QCCgWCymvXv3qqamRn6/3+PqRg/67g367g367g367o1M9T0Siai0tDT+e/xasi58DL7UEggE4uEjLy9PgUCAxWkQffcGffcGffcGffdGpvs+lLdM8IZTAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYNc7rAky7ceUur0tw7MRzD3tdAgAAruHMBwAAMIrwAQAAjCJ8AAAAowgfAADAqGGFj3Xr1snn82n58uXxfbZtKxQKqaSkRBMnTtTcuXN19OjR4dYJAABGiLTDx4EDB/Tiiy9q2rRpCfvXr1+vDRs2aNOmTTpw4ICKi4tVXV2tnp6eYRcLAAByX1rh4/z58/rWt76lX//61/rUpz4V32/btjZu3Kg1a9Zo4cKFqqys1NatW9XX16empibXigYAALkrrc/5WLp0qR5++GF95Stf0U9/+tP4/vb2dnV2dqqmpia+z7IszZkzR62trVqyZEnSXNFoVNFoNL4diUQkSbFYLH4Z3HaDNdZ2ZR6T3PrZ0zmmF8cezei7N+i7N+i7NzLVdyfzOQ4f27ZtU1tbmw4ePJh0XWdnpySpqKgoYX9RUZFOnjyZcr5169apoaEhaf/evXuVl5cX3w6Hw05LTWn9na5MY9Rrr73m2bHd6jucoe/eoO/eoO/ecLvvfX19Qx7rKHycOnVKP/jBD7R3715NmDDhquN8Pl/Ctm3bSfsGrVq1SvX19fHtSCSi0tJS1dTUKBAIKBaLKRwOq7q6Wn6/30m5KVWG9gx7DtPeDj1g/Jhu9x1DQ9+9Qd+9Qd+9kam+D75yMRSOwkdbW5u6uro0ffr0+L7+/n7t27dPmzZt0rFjxyT99wxIMBiMj+nq6ko6GzLIsixZlpW03+/3JzTlyu10RftTh6Bs5uWD0q2+wxn67g367g367g23++5kLkdvOL3//vt15MgRHT58OH6ZMWOGvvWtb+nw4cO66aabVFxcnHAq5+LFi2ppadHs2bOdHAoAAIxQjs585Ofnq7KyMmHfddddpxtuuCG+f/ny5WpsbFRFRYUqKirU2NiovLw81dXVuVc1AADIWa5/q+2KFSt04cIFPfnkkzp79qxmzpypvXv3Kj8/3+1DAQCAHDTs8NHc3Jyw7fP5FAqFFAqFhjs1AAAYgfhuFwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRjsLH5s2bNW3aNAUCAQUCAVVVVen111+PX7948WL5fL6Ey6xZs1wvGgAA5K5xTgZPnTpVzz33nD73uc9JkrZu3aqvfvWrOnTokL74xS9Kkh588EFt2bIlfpvx48e7WC4AAMh1jsLH/PnzE7afffZZbd68Wfv374+HD8uyVFxc7F6FAABgRHEUPi7X39+v3//+9+rt7VVVVVV8f3NzswoLC3X99ddrzpw5evbZZ1VYWHjVeaLRqKLRaHw7EolIkmKxWPwyuO0Ga6ztyjwmufWzp3NML449mtF3b9B3b9B3b2Sq707m89m27ei38ZEjR1RVVaWPP/5YkyZNUlNTkx566CFJ0vbt2zVp0iSVlZWpvb1da9eu1aVLl9TW1ibLslLOFwqF1NDQkLS/qalJeXl5TkoDAAAe6evrU11dnbq7uxUIBK451nH4uHjxot577z2dO3dOO3bs0G9+8xu1tLToC1/4QtLYjo4OlZWVadu2bVq4cGHK+VKd+SgtLdXp06cVCAQUi8UUDodVXV0tv9/vpNSUKkN7hj2HaW+HHjB+TLf7jqGh796g796g797IVN8jkYgKCgqGFD4cv+wyfvz4+BtOZ8yYoQMHDuj555/Xr371q6SxwWBQZWVlOn78+FXnsywr5VkRv9+f0JQrt9MV7fcNew7TvHxQutV3OEPfvUHfvUHfveF2353MNezP+bBtO+HMxeXOnDmjU6dOKRgMDvcwAABghHB05mP16tWqra1VaWmpenp6tG3bNjU3N2v37t06f/68QqGQHn30UQWDQZ04cUKrV69WQUGBHnnkkUzVDwAAcoyj8PHhhx/qscceU0dHhyZPnqxp06Zp9+7dqq6u1oULF3TkyBG9/PLLOnfunILBoObNm6ft27crPz8/U/UDAIAc4yh8vPTSS1e9buLEidqzJ/fezAkAAMziu10AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABglKOPVweA4bpx5S6vS0hijbW1/k6pMrRH0X5fyjEnnnvYcFXAyMWZDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGCUo/CxefNmTZs2TYFAQIFAQFVVVXr99dfj19u2rVAopJKSEk2cOFFz587V0aNHXS8aAADkLkfhY+rUqXruued08OBBHTx4UPfdd5+++tWvxgPG+vXrtWHDBm3atEkHDhxQcXGxqqur1dPTk5HiAQBA7nEUPubPn6+HHnpIN998s26++WY9++yzmjRpkvbv3y/btrVx40atWbNGCxcuVGVlpbZu3aq+vj41NTVlqn4AAJBjxqV7w/7+fv3+979Xb2+vqqqq1N7ers7OTtXU1MTHWJalOXPmqLW1VUuWLEk5TzQaVTQajW9HIhFJUiwWi18Gt91gjbVdmcckt372dI7pxbFHs9HQ92x8DFpj7IR/UxnJ94lXRsN6z0aZ6ruT+Xy2bTt6Jjhy5Iiqqqr08ccfa9KkSWpqatJDDz2k1tZW3XXXXXr//fdVUlISH//444/r5MmT2rNnT8r5QqGQGhoakvY3NTUpLy/PSWkAAMAjfX19qqurU3d3twKBwDXHOj7zccstt+jw4cM6d+6cduzYoUWLFqmlpSV+vc/nSxhv23bSvsutWrVK9fX18e1IJKLS0lLV1NQoEAgoFospHA6rurpafr/fablJKkOpQ1A2ezv0gPFjut13DM1o6Hs2PgatMbZ+MmNAaw+OUXQg9fOVF4/DkW40rPdslKm+D75yMRSOw8f48eP1uc99TpI0Y8YMHThwQM8//7yeeeYZSVJnZ6eCwWB8fFdXl4qKiq46n2VZsiwrab/f709oypXb6Yr2Xz0IZSsvH5Ru9R3OjOS+Z/NjMDrgu2p9I/X+yAYjeb1nM7f77mSuYX/Oh23bikajKi8vV3FxscLhcPy6ixcvqqWlRbNnzx7uYQAAwAjh6MzH6tWrVVtbq9LSUvX09Gjbtm1qbm7W7t275fP5tHz5cjU2NqqiokIVFRVqbGxUXl6e6urqMlU/AADIMY7Cx4cffqjHHntMHR0dmjx5sqZNm6bdu3erurpakrRixQpduHBBTz75pM6ePauZM2dq7969ys/Pz0jxAAAg9zgKHy+99NI1r/f5fAqFQgqFQsOpCQAAjGB8twsAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAoxyFj3Xr1umOO+5Qfn6+CgsLtWDBAh07dixhzOLFi+Xz+RIus2bNcrVoAACQuxyFj5aWFi1dulT79+9XOBzWpUuXVFNTo97e3oRxDz74oDo6OuKX1157zdWiAQBA7hrnZPDu3bsTtrds2aLCwkK1tbXp3nvvje+3LEvFxcXuVAgAAEaUYb3no7u7W5I0ZcqUhP3Nzc0qLCzUzTffrO9+97vq6uoazmEAAMAI4ujMx+Vs21Z9fb3uvvtuVVZWxvfX1tbq61//usrKytTe3q61a9fqvvvuU1tbmyzLSponGo0qGo3GtyORiCQpFovFL4PbbrDG2q7MY5JbP3s6x/Ti2KPZaOh7Nj4GrTF2wr+pjOT7xCujYb1no0z13cl8Ptu203omWLp0qXbt2qW33npLU6dOveq4jo4OlZWVadu2bVq4cGHS9aFQSA0NDUn7m5qalJeXl05pAADAsL6+PtXV1am7u1uBQOCaY9MKH8uWLdMrr7yiffv2qby8/BPHV1RU6Dvf+Y6eeeaZpOtSnfkoLS3V6dOnFQgEFIvFFA6HVV1dLb/f77TUJJWhPcOew7S3Qw8YP6bbfcfQjIa+Z+Nj0Bpj6yczBrT24BhFB3wpx3jxOBzpRsN6z0aZ6nskElFBQcGQwoejl11s29ayZcu0c+dONTc3Dyl4nDlzRqdOnVIwGEx5vWVZKV+O8fv9CU25cjtd0f7UTyzZzMsHpVt9hzMjue/Z/BiMDviuWt9IvT+ywUhe79nM7b47mcvRG06XLl2q3/3ud2pqalJ+fr46OzvV2dmpCxcuSJLOnz+vp59+Wn/961914sQJNTc3a/78+SooKNAjjzzi7KcAAAAjkqMzH5s3b5YkzZ07N2H/li1btHjxYo0dO1ZHjhzRyy+/rHPnzikYDGrevHnavn278vPzXSsaAADkLscvu1zLxIkTtWdP9r2eCwAAsgff7QIAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAoR9/tAgDIHTeu3OV1CddkjbW1/k6pMrRH0X6fJOnEcw97XBVM4MwHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwChH4WPdunW64447lJ+fr8LCQi1YsEDHjh1LGGPbtkKhkEpKSjRx4kTNnTtXR48edbVoAACQuxyFj5aWFi1dulT79+9XOBzWpUuXVFNTo97e3viY9evXa8OGDdq0aZMOHDig4uJiVVdXq6enx/XiAQBA7hnnZPDu3bsTtrds2aLCwkK1tbXp3nvvlW3b2rhxo9asWaOFCxdKkrZu3aqioiI1NTVpyZIl7lUOAABy0rDe89Hd3S1JmjJliiSpvb1dnZ2dqqmpiY+xLEtz5sxRa2vrcA4FAABGCEdnPi5n27bq6+t19913q7KyUpLU2dkpSSoqKkoYW1RUpJMnT6acJxqNKhqNxrcjkYgkKRaLxS+D226wxtquzGOSWz97Osf04tij2WjoezY+Bq0xdsK/qeTifZKNvb5cqr7nYp9zTaaeZ5zM57NtO63VuXTpUu3atUtvvfWWpk6dKklqbW3VXXfdpQ8++EDBYDA+9rvf/a5OnTqV9LKNJIVCITU0NCTtb2pqUl5eXjqlAQAAw/r6+lRXV6fu7m4FAoFrjk3rzMeyZcv06quvat++ffHgIUnFxcWS/nsG5PLw0dXVlXQ2ZNCqVatUX18f345EIiotLVVNTY0CgYBisZjC4bCqq6vl9/vTKTdBZWjPsOcw7e3QA8aP6XbfMTSjoe/Z+Bi0xtj6yYwBrT04RtEBX8oxXjwOhysbe325VH3PxT7nmkw9zwy+cjEUjsKHbdtatmyZdu7cqebmZpWXlydcX15eruLiYoXDYd1+++2SpIsXL6qlpUU/+9nPUs5pWZYsy0ra7/f7E5py5Xa6ov2pn1iymZe/hNzqO5wZyX3P5sdgdMB31fpy8f7I5l5f7vK+52Kfc5XbzzNO5nIUPpYuXaqmpib96U9/Un5+fvw9HpMnT9bEiRPl8/m0fPlyNTY2qqKiQhUVFWpsbFReXp7q6uqc/RQAAGBEchQ+Nm/eLEmaO3duwv4tW7Zo8eLFkqQVK1bowoULevLJJ3X27FnNnDlTe/fuVX5+visFAwCA3Ob4ZZdP4vP5FAqFFAqF0q0JAACMYHy3CwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAox+Fj3759mj9/vkpKSuTz+fTKK68kXL948WL5fL6Ey6xZs9yqFwAA5DjH4aO3t1e33XabNm3adNUxDz74oDo6OuKX1157bVhFAgCAkWOc0xvU1taqtrb2mmMsy1JxcXHaRQEAgJHLcfgYiubmZhUWFur666/XnDlz9Oyzz6qwsDDl2Gg0qmg0Gt+ORCKSpFgsFr8MbrvBGmu7Mo9Jbv3s6RzTi2OPZqOh79n4GLTG2An/ppKL90k29vpyqfqei33ONZl6nnEyn8+27bRXp8/n086dO7VgwYL4vu3bt2vSpEkqKytTe3u71q5dq0uXLqmtrU2WZSXNEQqF1NDQkLS/qalJeXl56ZYGAAAM6uvrU11dnbq7uxUIBK451vXwcaWOjg6VlZVp27ZtWrhwYdL1qc58lJaW6vTp0woEAorFYgqHw6qurpbf70+31LjK0J5hz2Ha26EHjB/T7b5jaEZD37PxMWiNsfWTGQNae3CMogO+lGO8eBwOVzb2+nKp+p6Lfc41mXqeiUQiKigoGFL4yMjLLpcLBoMqKyvT8ePHU15vWVbKMyJ+vz+hKVdupyvan/qJJZt5+UvIrb7DmZHc92x+DEYHfFetLxfvj2zu9eUu73su9jlXuf0842SujH/Ox5kzZ3Tq1CkFg8FMHwoAAOQAx2c+zp8/r3fffTe+3d7ersOHD2vKlCmaMmWKQqGQHn30UQWDQZ04cUKrV69WQUGBHnnkEVcLBwAAuclx+Dh48KDmzZsX366vr5ckLVq0SJs3b9aRI0f08ssv69y5cwoGg5o3b562b9+u/Px896oGAAA5y3H4mDt3rq71HtU9e7L7DU4AAMBbfLcLAAAwivABAACMyvif2gIAMJLduHKX1yU4Yo21tf5Ob2vgzAcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMMpx+Ni3b5/mz5+vkpIS+Xw+vfLKKwnX27atUCikkpISTZw4UXPnztXRo0fdqhcAAOQ4x+Gjt7dXt912mzZt2pTy+vXr12vDhg3atGmTDhw4oOLiYlVXV6unp2fYxQIAgNw3zukNamtrVVtbm/I627a1ceNGrVmzRgsXLpQkbd26VUVFRWpqatKSJUuGVy0AAMh5jsPHtbS3t6uzs1M1NTXxfZZlac6cOWptbU0ZPqLRqKLRaHw7EolIkmKxWPwyuO0Ga6ztyjwmufWzp3NML449mo2GvmfjY9AaYyf8m0ou3ifZ2OvLpeo7fc68wX673Wsn8/ls2067az6fTzt37tSCBQskSa2trbrrrrv0/vvvq6SkJD7u8ccf18mTJ7Vnz56kOUKhkBoaGpL2NzU1KS8vL93SAACAQX19faqrq1N3d7cCgcA1x7p65mOQz+dL2LZtO2nfoFWrVqm+vj6+HYlEVFpaqpqaGgUCAcViMYXDYVVXV8vv9w+7tspQcgDKdm+HHjB+TLf7jqEZDX3PxsegNcbWT2YMaO3BMYoOpH6u8uJxOFzZ2OvLpeo7fc68wb67/Twz+MrFULgaPoqLiyVJnZ2dCgaD8f1dXV0qKipKeRvLsmRZVtJ+v9+f0JQrt9MV7U/9xJLNvPwl5Fbf4cxI7ns2PwajA76r1peL90c29/pyl/edPpvj9vOMk7lc/ZyP8vJyFRcXKxwOx/ddvHhRLS0tmj17tpuHAgAAOcrxmY/z58/r3XffjW+3t7fr8OHDmjJlij7zmc9o+fLlamxsVEVFhSoqKtTY2Ki8vDzV1dW5WjgAAMhNjsPHwYMHNW/evPj24Ps1Fi1apN/+9rdasWKFLly4oCeffFJnz57VzJkztXfvXuXn57tXNQAAyFmOw8fcuXN1rT+Q8fl8CoVCCoVCw6kLAACMUHy3CwAAMIrwAQAAjMrI53wAAJCOG1fu8roEGMCZDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRfMhYDvDiQ3essbbW3ylVhvYo2u9zfPsTzz2cgapwJT6QCUAu4swHAAAwivABAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKP4nA/gf7LhMzOG+/kqAJALOPMBAACMInwAAACjCB8AAMAowgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIoPGQOAIciGD6EDRgrOfAAAAKMIHwAAwCjCBwAAMMr18BEKheTz+RIuxcXFbh8GAADkqIy84fSLX/yi/vznP8e3x44dm4nDAACAHJSR8DFu3DjOdgAAgJQyEj6OHz+ukpISWZalmTNnqrGxUTfddFPKsdFoVNFoNL4diUQkSbFYLH4Z3HaDNdZ2ZZ6RzhpjJ/zrlFv3l0nZsDaG23ekh757g757Y7Dfbj9PO5nPZ9u2q/f666+/rr6+Pt1888368MMP9dOf/lTvvPOOjh49qhtuuCFpfCgUUkNDQ9L+pqYm5eXluVkaAADIkL6+PtXV1am7u1uBQOCaY10PH1fq7e3VZz/7Wa1YsUL19fVJ16c681FaWqrTp08rEAgoFospHA6rurpafr9/2PVUhvYMe47RwBpj6yczBrT24BhFB3yOb/926IEMVJVZ2bA2htt3pIe+e4O+e2Ow7279Xh0UiURUUFAwpPCR8U84ve666/SlL31Jx48fT3m9ZVmyLCtpv9/vT2jKldvpivazwJ2IDvjS6pmbC9qUbFob6fYdw0PfvUHfveHW79XL5xuqjH/ORzQa1T//+U8Fg8FMHwoAAOQA18PH008/rZaWFrW3t+tvf/ubvva1rykSiWjRokVuHwoAAOQg1192+c9//qNvfvObOn36tD796U9r1qxZ2r9/v8rKytw+FAAAyEGuh49t27a5PSUAABhB+G4XAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBRhA8AAGAU4QMAABhF+AAAAEYRPgAAgFGEDwAAYBThAwAAGEX4AAAARhE+AACAUYQPAABgFOEDAAAYRfgAAABGET4AAIBR47wuACPTjSt3eV0CACBLceYDAAAYRfgAAABGET4AAIBRGQsfL7zwgsrLyzVhwgRNnz5db775ZqYOBQAAckhGwsf27du1fPlyrVmzRocOHdI999yj2tpavffee5k4HAAAyCEZCR8bNmzQt7/9bX3nO9/R5z//eW3cuFGlpaXavHlzJg4HAAByiOt/anvx4kW1tbVp5cqVCftramrU2tqaND4ajSoajca3u7u7JUkfffSRYrGYYrGY+vr6dObMGfn9/mHXN+5S77DnGA3GDdjq6xvQuNgY9Q/4vC5n1KDv3qDv3qDv3hjsu1u/Vwf19PRIkmzb/uQaXDvq/5w+fVr9/f0qKipK2F9UVKTOzs6k8evWrVNDQ0PS/vLycrdLg0N1XhcwStF3b9B3b9B3b2Sy7z09PZo8efI1x2TsQ8Z8vsQUa9t20j5JWrVqlerr6+PbAwMD+uijj3TDDTfI5/MpEomotLRUp06dUiAQyFS5uAJ99wZ99wZ99wZ990am+m7btnp6elRSUvKJY10PHwUFBRo7dmzSWY6urq6ksyGSZFmWLMtK2Hf99dcnjQsEAixOD9B3b9B3b9B3b9B3b2Si7590xmOQ6284HT9+vKZPn65wOJywPxwOa/bs2W4fDgAA5JiMvOxSX1+vxx57TDNmzFBVVZVefPFFvffee3riiScycTgAAJBDMhI+vvGNb+jMmTP68Y9/rI6ODlVWVuq1115TWVmZ47ksy9KPfvSjpJdmkFn03Rv03Rv03Rv03RvZ0HefPZS/iQEAAHAJ3+0CAACMInwAAACjCB8AAMAowgcAADAqK8LHCy+8oPLyck2YMEHTp0/Xm2++ec3xLS0tmj59uiZMmKCbbrpJv/zlLw1VOrI46Xtzc7N8Pl/S5Z133jFYce7bt2+f5s+fr5KSEvl8Pr3yyiufeBvW+/A57TvrffjWrVunO+64Q/n5+SosLNSCBQt07NixT7wd63140um7F+vd8/Cxfft2LV++XGvWrNGhQ4d0zz33qLa2Vu+9917K8e3t7XrooYd0zz336NChQ1q9erW+//3va8eOHYYrz21O+z7o2LFj6ujoiF8qKioMVTwy9Pb26rbbbtOmTZuGNJ717g6nfR/Eek9fS0uLli5dqv379yscDuvSpUuqqalRb+/Vv9yT9T586fR9kNH1bnvszjvvtJ944omEfbfeequ9cuXKlONXrFhh33rrrQn7lixZYs+aNStjNY5ETvv+xhtv2JLss2fPGqhudJBk79y585pjWO/uG0rfWe/u6+rqsiXZLS0tVx3DenffUPruxXr39MzHxYsX1dbWppqamoT9NTU1am1tTXmbv/71r0njH3jgAR08eFCxWCxjtY4k6fR90O23365gMKj7779fb7zxRibLhFjvXmO9u6e7u1uSNGXKlKuOYb27byh9H2RyvXsaPk6fPq3+/v6kL5wrKipK+mK6QZ2dnSnHX7p0SadPn85YrSNJOn0PBoN68cUXtWPHDv3xj3/ULbfcovvvv1/79u0zUfKoxXr3BuvdXbZtq76+XnfffbcqKyuvOo717q6h9t2L9Z6Rj1d3yufzJWzbtp2075PGp9qPa3PS91tuuUW33HJLfLuqqkqnTp3S//3f/+nee+/NaJ2jHevdPNa7u5566in9/e9/11tvvfWJY1nv7hlq371Y756e+SgoKNDYsWOT/rfd1dWVlH4HFRcXpxw/btw43XDDDRmrdSRJp++pzJo1S8ePH3e7PFyG9Z49WO/pWbZsmV599VW98cYbmjp16jXHst7d46TvqWR6vXsaPsaPH6/p06crHA4n7A+Hw5o9e3bK21RVVSWN37t3r2bMmCG/35+xWkeSdPqeyqFDhxQMBt0uD5dhvWcP1rsztm3rqaee0h//+Ef95S9/UXl5+SfehvU+fOn0PZWMr3djb229im3bttl+v99+6aWX7H/84x/28uXL7euuu84+ceKEbdu2vXLlSvuxxx6Lj//3v/9t5+Xl2T/84Q/tf/zjH/ZLL71k+/1++w9/+INXP0JOctr3n//85/bOnTvtf/3rX/bbb79tr1y50pZk79ixw6sfISf19PTYhw4dsg8dOmRLsjds2GAfOnTIPnnypG3brPdMcdp31vvwfe9737MnT55sNzc32x0dHfFLX19ffAzr3X3p9N2L9e55+LBt2/7FL35hl5WV2ePHj7e//OUvJ/xJ0KJFi+w5c+YkjG9ubrZvv/12e/z48faNN95ob9682XDFI4OTvv/sZz+zP/vZz9oTJkywP/WpT9l33323vWvXLg+qzm2Df9J25WXRokW2bbPeM8Vp31nvw5eq35LsLVu2xMew3t2XTt+9WO++/xULAABghOefcAoAAEYXwgcAADCK8AEAAIwifAAAAKMIHwAAwCjCBwAAMIrwAQAAjCJ8AAAAowgfAADAKMIHAAAwivABAACMInwAAACj/h9gwvhFnSid9QAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['petal_width'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "f4cea2a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sepal_length</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.109369</td>\n",
       "      <td>0.871754</td>\n",
       "      <td>0.817954</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sepal_width</th>\n",
       "      <td>-0.109369</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.420516</td>\n",
       "      <td>-0.356544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal_length</th>\n",
       "      <td>0.871754</td>\n",
       "      <td>-0.420516</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.962757</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal_width</th>\n",
       "      <td>0.817954</td>\n",
       "      <td>-0.356544</td>\n",
       "      <td>0.962757</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              sepal_length  sepal_width  petal_length  petal_width\n",
       "sepal_length      1.000000    -0.109369      0.871754     0.817954\n",
       "sepal_width      -0.109369     1.000000     -0.420516    -0.356544\n",
       "petal_length      0.871754    -0.420516      1.000000     0.962757\n",
       "petal_width       0.817954    -0.356544      0.962757     1.000000"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "8c998eb0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 500x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "corr = data.corr()\n",
    "fig,ax = plt.subplots(figsize=(5,4))\n",
    "sns.heatmap(corr,annot=True,ax=ax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "cf81da18",
   "metadata": {},
   "outputs": [],
   "source": [
    "#label encoder\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "05c887ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal_length  sepal_width  petal_length  petal_width  species\n",
       "0           5.1          3.5           1.4          0.2        0\n",
       "1           4.9          3.0           1.4          0.2        0\n",
       "2           4.7          3.2           1.3          0.2        0\n",
       "3           4.6          3.1           1.5          0.2        0\n",
       "4           5.0          3.6           1.4          0.2        0"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['species'] = le.fit_transform(data['species'])\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "7ca1ff24",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model training \n",
    "from sklearn.model_selection import train_test_split\n",
    "#train-70%\n",
    "#test-30%\n",
    "x=data.drop(columns=['species'])\n",
    "y=data['species']\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "1472cf16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic Regrssion \n",
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "9b256301",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "431b61a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 93.33333333333333\n"
     ]
    }
   ],
   "source": [
    "# print matric to get performance\n",
    "print(\"Accuracy:\",model.score(x_test,y_test)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "bb1357b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import K nearest neighbours \n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "model = KNeighborsClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "313e1615",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "f1b6beea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 91.11111111111111\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\neighbors\\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.\n",
      "  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy:\",model.score(x_test,y_test)*100)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
