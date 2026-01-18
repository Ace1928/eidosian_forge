import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _power_matrix(self, a, n):
    if sparse.issparse(a):
        a = a ** n
    else:
        a = np.linalg.matrix_power(a, n)
    return a