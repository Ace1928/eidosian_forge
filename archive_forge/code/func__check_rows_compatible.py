from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _check_rows_compatible(*data):
    for d in data:
        if not _get_row_length(d) == _get_row_length(data[0]):
            raise ValueError('Expected `data` and `extra_data` to have the same number of rows. Got {}'.format([d.shape[0] for d in data]))
        if isinstance(d, (pd.DataFrame, pd.Series)) and isinstance(data[0], (pd.DataFrame, pd.Series)):
            if not np.all(data[0].index == d.index):
                raise ValueError('Expected `data` and `extra_data` pandas inputs to have the same index. Fix with `scprep.select.select_rows(*extra_data, idx=data.index)`')