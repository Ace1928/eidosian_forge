from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def SparseDataFrame(X, columns=None, index=None, default_fill_value=0.0):
    if sparse.issparse(X):
        X = pd.DataFrame.sparse.from_spmatrix(X)
        X.sparse.fill_value = default_fill_value
    else:
        if is_SparseDataFrame(X) or not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = dataframe_to_sparse(X, fill_value=default_fill_value)
    if columns is not None:
        X.columns = columns
    if index is not None:
        X.index = index
    return X