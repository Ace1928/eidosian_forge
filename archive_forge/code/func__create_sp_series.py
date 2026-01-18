from datetime import timedelta
import os
import pickle
import platform as pl
import sys
import numpy as np
import pandas
from pandas import (
from pandas.arrays import SparseArray
from pandas.tseries.offsets import (
def _create_sp_series():
    nan = np.nan
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan
    bseries = Series(SparseArray(arr, kind='block'))
    bseries.name = 'bseries'
    return bseries