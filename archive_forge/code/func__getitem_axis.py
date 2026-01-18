from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
def _getitem_axis(self, key, axis: AxisInt):
    if key is Ellipsis:
        key = slice(None)
    elif isinstance(key, ABCDataFrame):
        raise IndexError('DataFrame indexer is not allowed for .iloc\nConsider using .loc for automatic alignment.')
    if isinstance(key, slice):
        return self._get_slice_axis(key, axis=axis)
    if is_iterator(key):
        key = list(key)
    if isinstance(key, list):
        key = np.asarray(key)
    if com.is_bool_indexer(key):
        self._validate_key(key, axis)
        return self._getbool_axis(key, axis=axis)
    elif is_list_like_indexer(key):
        return self._get_list_axis(key, axis=axis)
    else:
        key = item_from_zerodim(key)
        if not is_integer(key):
            raise TypeError('Cannot index by location index with a non-integer key')
        self._validate_integer(key, axis)
        return self.obj._ixs(key, axis=axis)