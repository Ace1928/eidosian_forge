from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _normalize_dataframe(dataframe, index):
    """Take a pandas DataFrame and count the element present in the
    given columns, return a hierarchical index on those columns
    """
    data = dataframe[index].dropna()
    grouped = data.groupby(index, sort=False, observed=False)
    counted = grouped[index].count()
    averaged = counted.mean(axis=1)
    averaged = averaged.fillna(0.0)
    return averaged