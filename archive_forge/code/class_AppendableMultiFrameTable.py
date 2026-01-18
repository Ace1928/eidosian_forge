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
class AppendableMultiFrameTable(AppendableFrameTable):
    """a frame with a multi-index"""
    table_type = 'appendable_multiframe'
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile('^level_\\d+$')

    @property
    def table_type_short(self) -> str:
        return 'appendable_multi'

    def write(self, obj, data_columns=None, **kwargs) -> None:
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(self, where=None, columns=None, start: int | None=None, stop: int | None=None):
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)
        df.index = df.index.set_names([None if self._re_levels.search(name) else name for name in df.index.names])
        return df