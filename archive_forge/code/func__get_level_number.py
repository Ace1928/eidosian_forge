from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def _get_level_number(self, level) -> int:
    count = self.names.count(level)
    if count > 1 and (not is_integer(level)):
        raise ValueError(f'The name {level} occurs multiple times, use a level number')
    try:
        level = self.names.index(level)
    except ValueError as err:
        if not is_integer(level):
            raise KeyError(f'Level {level} not found') from err
        if level < 0:
            level += self.nlevels
            if level < 0:
                orig_level = level - self.nlevels
                raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, {orig_level} is not a valid level number') from err
        elif level >= self.nlevels:
            raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, not {level + 1}') from err
    return level