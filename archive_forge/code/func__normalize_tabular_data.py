from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _normalize_tabular_data(tabular_data, headers, showindex='default'):
    """Transform a supported data type to a list of lists, and a list of headers.

    Supported tabular data types:

    * list-of-lists or another iterable of iterables

    * list of named tuples (usually used with headers="keys")

    * list of dicts (usually used with headers="keys")

    * list of OrderedDicts (usually used with headers="keys")

    * list of dataclasses (Python 3.7+ only, usually used with headers="keys")

    * 2D NumPy arrays

    * NumPy record arrays (usually used with headers="keys")

    * dict of iterables (usually used with headers="keys")

    * pandas.DataFrame (usually used with headers="keys")

    The first row can be used as headers if headers="firstrow",
    column indices can be used as headers if headers="keys".

    If showindex="default", show row indices of the pandas.DataFrame.
    If showindex="always", show row indices for all types of data.
    If showindex="never", don't show row indices for all types of data.
    If showindex is an iterable, show its values as row indices.

    """
    try:
        bool(headers)
        is_headers2bool_broken = False
    except ValueError:
        is_headers2bool_broken = True
        headers = list(headers)
    index = None
    if hasattr(tabular_data, 'keys') and hasattr(tabular_data, 'values'):
        if hasattr(tabular_data.values, '__call__'):
            keys = tabular_data.keys()
            rows = list(izip_longest(*tabular_data.values()))
        elif hasattr(tabular_data, 'index'):
            keys = list(tabular_data)
            if showindex in ['default', 'always', True] and tabular_data.index.name is not None:
                if isinstance(tabular_data.index.name, list):
                    keys[:0] = tabular_data.index.name
                else:
                    keys[:0] = [tabular_data.index.name]
            vals = tabular_data.values
            index = list(tabular_data.index)
            rows = [list(row) for row in vals]
        else:
            raise ValueError("tabular data doesn't appear to be a dict or a DataFrame")
        if headers == 'keys':
            headers = list(map(str, keys))
    else:
        rows = list(tabular_data)
        if headers == 'keys' and (not rows):
            headers = []
        elif headers == 'keys' and hasattr(tabular_data, 'dtype') and getattr(tabular_data.dtype, 'names'):
            headers = tabular_data.dtype.names
        elif headers == 'keys' and len(rows) > 0 and isinstance(rows[0], tuple) and hasattr(rows[0], '_fields'):
            headers = list(map(str, rows[0]._fields))
        elif len(rows) > 0 and hasattr(rows[0], 'keys') and hasattr(rows[0], 'values'):
            uniq_keys = set()
            keys = []
            if headers == 'firstrow':
                firstdict = rows[0] if len(rows) > 0 else {}
                keys.extend(firstdict.keys())
                uniq_keys.update(keys)
                rows = rows[1:]
            for row in rows:
                for k in row.keys():
                    if k not in uniq_keys:
                        keys.append(k)
                        uniq_keys.add(k)
            if headers == 'keys':
                headers = keys
            elif isinstance(headers, dict):
                headers = [headers.get(k, k) for k in keys]
                headers = list(map(str, headers))
            elif headers == 'firstrow':
                if len(rows) > 0:
                    headers = [firstdict.get(k, k) for k in keys]
                    headers = list(map(str, headers))
                else:
                    headers = []
            elif headers:
                raise ValueError('headers for a list of dicts is not a dict or a keyword')
            rows = [[row.get(k) for k in keys] for row in rows]
        elif headers == 'keys' and hasattr(tabular_data, 'description') and hasattr(tabular_data, 'fetchone') and hasattr(tabular_data, 'rowcount'):
            headers = [column[0] for column in tabular_data.description]
        elif dataclasses is not None and len(rows) > 0 and dataclasses.is_dataclass(rows[0]):
            field_names = [field.name for field in dataclasses.fields(rows[0])]
            if headers == 'keys':
                headers = field_names
            rows = [[getattr(row, f) for f in field_names] for row in rows]
        elif headers == 'keys' and len(rows) > 0:
            headers = list(map(str, range(len(rows[0]))))
    if headers == 'firstrow' and len(rows) > 0:
        if index is not None:
            headers = [index[0]] + list(rows[0])
            index = index[1:]
        else:
            headers = rows[0]
        headers = list(map(str, headers))
        rows = rows[1:]
    elif headers == 'firstrow':
        headers = []
    headers = list(map(str, headers))
    rows = list(map(lambda r: r if _is_separating_line(r) else list(r), rows))
    showindex_is_a_str = type(showindex) in [str, bytes]
    if showindex == 'default' and index is not None:
        rows = _prepend_row_index(rows, index)
    elif isinstance(showindex, Sized) and (not showindex_is_a_str):
        rows = _prepend_row_index(rows, list(showindex))
    elif isinstance(showindex, Iterable) and (not showindex_is_a_str):
        rows = _prepend_row_index(rows, showindex)
    elif showindex == 'always' or (_bool(showindex) and (not showindex_is_a_str)):
        if index is None:
            index = list(range(len(rows)))
        rows = _prepend_row_index(rows, index)
    elif showindex == 'never' or (not _bool(showindex) and (not showindex_is_a_str)):
        pass
    if headers and len(rows) > 0:
        nhs = len(headers)
        ncols = len(rows[0])
        if nhs < ncols:
            headers = [''] * (ncols - nhs) + headers
    return (rows, headers)