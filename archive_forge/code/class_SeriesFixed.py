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
class SeriesFixed(GenericFixed):
    pandas_kind = 'series'
    attributes = ['name']
    name: Hashable

    @property
    def shape(self):
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(self, where=None, columns=None, start: int | None=None, stop: int | None=None) -> Series:
        self.validate_read(columns, where)
        index = self.read_index('index', start=start, stop=stop)
        values = self.read_array('values', start=start, stop=stop)
        result = Series(values, index=index, name=self.name, copy=False)
        if using_pyarrow_string_dtype() and is_string_array(values, skipna=True):
            result = result.astype('string[pyarrow_numpy]')
        return result

    def write(self, obj, **kwargs) -> None:
        super().write(obj, **kwargs)
        self.write_index('index', obj.index)
        self.write_array('values', obj)
        self.attrs.name = obj.name