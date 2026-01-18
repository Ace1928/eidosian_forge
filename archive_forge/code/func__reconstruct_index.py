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
def _reconstruct_index(table, index_descriptors, all_columns, types_mapper=None):
    field_name_to_metadata = {c.get('field_name', c['name']): c for c in all_columns}
    index_arrays = []
    index_names = []
    result_table = table
    for descr in index_descriptors:
        if isinstance(descr, str):
            result_table, index_level, index_name = _extract_index_level(table, result_table, descr, field_name_to_metadata, types_mapper)
            if index_level is None:
                continue
        elif descr['kind'] == 'range':
            index_name = descr['name']
            index_level = _pandas_api.pd.RangeIndex(descr['start'], descr['stop'], step=descr['step'], name=index_name)
            if len(index_level) != len(table):
                continue
        else:
            raise ValueError('Unrecognized index kind: {}'.format(descr['kind']))
        index_arrays.append(index_level)
        index_names.append(index_name)
    pd = _pandas_api.pd
    if len(index_arrays) > 1:
        index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
    elif len(index_arrays) == 1:
        index = index_arrays[0]
        if not isinstance(index, pd.Index):
            index = pd.Index(index, name=index_names[0])
    else:
        index = pd.RangeIndex(table.num_rows)
    return (result_table, index)