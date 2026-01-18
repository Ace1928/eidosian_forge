from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def _vector_coerce_two_dense(x, y):
    try:
        x = _vector_coerce_dense(x)
        y = _vector_coerce_dense(y)
    except ValueError as e:
        if 'x must be a 1D array. Got shape ' in str(e):
            raise ValueError('Expected x and y to be 1D arrays. Got shapes x {}, y {}'.format(x.shape, y.shape))
        else:
            raise e
    return (x, y)