from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    assert isinstance(name, str)
    index_name = index.name
    converted, dtype_name = _get_data_and_dtype_name(index)
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)
    if lib.is_np_dtype(index.dtype, 'iu') or needs_i8_conversion(index.dtype) or is_bool_dtype(index.dtype):
        return IndexCol(name, values=converted, kind=kind, typ=atom, freq=getattr(index, 'freq', None), tz=getattr(index, 'tz', None), index_name=index_name)
    if isinstance(index, MultiIndex):
        raise TypeError('MultiIndex not supported here!')
    inferred_type = lib.infer_dtype(index, skipna=False)
    values = np.asarray(index)
    if inferred_type == 'date':
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(name, converted, 'date', _tables().Time32Col(), index_name=index_name)
    elif inferred_type == 'string':
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(name, converted, 'string', _tables().StringCol(itemsize), index_name=index_name)
    elif inferred_type in ['integer', 'floating']:
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == 'object', kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)