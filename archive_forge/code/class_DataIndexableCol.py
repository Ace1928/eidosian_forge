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
class DataIndexableCol(DataCol):
    """represent a data column that can be indexed"""
    is_data_indexable = True

    def validate_names(self) -> None:
        if not is_string_dtype(Index(self.values).dtype):
            raise ValueError('cannot have non-object label DataIndexableCol')

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col()