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
@final
def _getitem_nested_tuple(self, tup: tuple):

    def _contains_slice(x: object) -> bool:
        if isinstance(x, tuple):
            return any((isinstance(v, slice) for v in x))
        elif isinstance(x, slice):
            return True
        return False
    for key in tup:
        check_dict_or_set_indexers(key)
    if len(tup) > self.ndim:
        if self.name != 'loc':
            raise ValueError('Too many indices')
        if all((is_hashable(x) and (not _contains_slice(x)) or com.is_null_slice(x) for x in tup)):
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
        elif isinstance(self.obj, ABCSeries) and any((isinstance(k, tuple) for k in tup)):
            raise IndexingError('Too many indexers')
        axis = self.axis or 0
        return self._getitem_axis(tup, axis=axis)
    obj = self.obj
    axis = len(tup) - 1
    for key in tup[::-1]:
        if com.is_null_slice(key):
            axis -= 1
            continue
        obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
        axis -= 1
        if is_scalar(obj) or not hasattr(obj, 'ndim'):
            break
    return obj