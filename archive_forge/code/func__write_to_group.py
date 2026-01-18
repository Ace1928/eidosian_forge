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
def _write_to_group(self, key: str, value: DataFrame | Series, format, axes=None, index: bool | list[str]=True, append: bool=False, complib=None, complevel: int | None=None, fletcher32=None, min_itemsize: int | dict[str, int] | None=None, chunksize: int | None=None, expectedrows=None, dropna: bool=False, nan_rep=None, data_columns=None, encoding=None, errors: str='strict', track_times: bool=True) -> None:
    if getattr(value, 'empty', None) and (format == 'table' or append):
        return
    group = self._identify_group(key, append)
    s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
    if append:
        if not s.is_table or (s.is_table and format == 'fixed' and s.is_exists):
            raise ValueError('Can only append to Tables')
        if not s.is_exists:
            s.set_object_info()
    else:
        s.set_object_info()
    if not s.is_table and complib:
        raise ValueError('Compression not supported on Fixed format stores')
    s.write(obj=value, axes=axes, append=append, complib=complib, complevel=complevel, fletcher32=fletcher32, min_itemsize=min_itemsize, chunksize=chunksize, expectedrows=expectedrows, dropna=dropna, nan_rep=nan_rep, data_columns=data_columns, track_times=track_times)
    if isinstance(s, Table) and index:
        s.create_index(columns=index)