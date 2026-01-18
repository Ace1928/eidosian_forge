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
def _reconstruct_block(item, columns=None, extension_columns=None):
    """
    Construct a pandas Block from the `item` dictionary coming from pyarrow's
    serialization or returned by arrow::python::ConvertTableToPandas.

    This function takes care of converting dictionary types to pandas
    categorical, Timestamp-with-timezones to the proper pandas Block, and
    conversion to pandas ExtensionBlock

    Parameters
    ----------
    item : dict
        For basic types, this is a dictionary in the form of
        {'block': np.ndarray of values, 'placement': pandas block placement}.
        Additional keys are present for other types (dictionary, timezone,
        object).
    columns :
        Column names of the table being constructed, used for extension types
    extension_columns : dict
        Dictionary of {column_name: pandas_dtype} that includes all columns
        and corresponding dtypes that will be converted to a pandas
        ExtensionBlock.

    Returns
    -------
    pandas Block

    """
    import pandas.core.internals as _int
    block_arr = item.get('block', None)
    placement = item['placement']
    if 'dictionary' in item:
        cat = _pandas_api.categorical_type.from_codes(block_arr, categories=item['dictionary'], ordered=item['ordered'])
        block = _int.make_block(cat, placement=placement)
    elif 'timezone' in item:
        unit, _ = np.datetime_data(block_arr.dtype)
        dtype = make_datetimetz(unit, item['timezone'])
        if _pandas_api.is_ge_v21():
            pd_arr = _pandas_api.pd.array(block_arr.view('int64'), dtype=dtype, copy=False)
            block = _int.make_block(pd_arr, placement=placement)
        else:
            block = _int.make_block(block_arr, placement=placement, klass=_int.DatetimeTZBlock, dtype=dtype)
    elif 'py_array' in item:
        arr = item['py_array']
        assert len(placement) == 1
        name = columns[placement[0]]
        pandas_dtype = extension_columns[name]
        if not hasattr(pandas_dtype, '__from_arrow__'):
            raise ValueError('This column does not support to be converted to a pandas ExtensionArray')
        pd_ext_arr = pandas_dtype.__from_arrow__(arr)
        block = _int.make_block(pd_ext_arr, placement=placement)
    else:
        block = _int.make_block(block_arr, placement=placement)
    return block