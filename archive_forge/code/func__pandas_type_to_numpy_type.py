import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _pandas_type_to_numpy_type(pandas_type):
    """Get the numpy dtype that corresponds to a pandas type.

    Parameters
    ----------
    pandas_type : str
        The result of a call to pandas.lib.infer_dtype.

    Returns
    -------
    dtype : np.dtype
        The dtype that corresponds to `pandas_type`.
    """
    try:
        return _pandas_logical_type_map[pandas_type]
    except KeyError:
        if 'mixed' in pandas_type:
            return np.object_
        return np.dtype(pandas_type)