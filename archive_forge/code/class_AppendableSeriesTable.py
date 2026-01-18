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
class AppendableSeriesTable(AppendableFrameTable):
    """support the new appendable table formats"""
    pandas_kind = 'series_table'
    table_type = 'appendable_series'
    ndim = 2
    obj_type = Series

    @property
    def is_transposed(self) -> bool:
        return False

    @classmethod
    def get_object(cls, obj, transposed: bool):
        return obj

    def write(self, obj, data_columns=None, **kwargs) -> None:
        """we are going to write this as a frame table"""
        if not isinstance(obj, DataFrame):
            name = obj.name or 'values'
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(self, where=None, columns=None, start: int | None=None, stop: int | None=None) -> Series:
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)
        s = s.iloc[:, 0]
        if s.name == 'values':
            s.name = None
        return s