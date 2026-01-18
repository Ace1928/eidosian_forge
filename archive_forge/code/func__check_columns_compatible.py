from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _check_columns_compatible(*data):
    for d in data:
        if not _get_column_length(d) == _get_column_length(data[0]):
            raise ValueError('Expected `data` and `extra_data` to have the same number of columns. Got {}'.format([_get_column_length(d) for d in data]))
        if isinstance(d, (pd.DataFrame, pd.Series)) and isinstance(data[0], (pd.DataFrame, pd.Series)):
            if not np.all(_get_columns(data[0]) == _get_columns(d)):
                raise ValueError('Expected `data` and `extra_data` pandas inputs to have the same column names. Fix with `scprep.select.select_cols(*extra_data, idx=data.columns)`')