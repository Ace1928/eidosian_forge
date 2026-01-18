import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
def add_attr(df, timestamp):
    if 'bin_bounds' in df.attrs:
        df.attrs['bin_bounds'] = (*df.attrs['bin_bounds'], timestamp)
    else:
        df.attrs['bin_bounds'] = (timestamp,)
    return df