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
def _setitem_with_indexer_frame_value(self, indexer, value: DataFrame, name: str):
    ilocs = self._ensure_iterable_column_indexer(indexer[1])
    sub_indexer = list(indexer)
    pi = indexer[0]
    multiindex_indexer = isinstance(self.obj.columns, MultiIndex)
    unique_cols = value.columns.is_unique
    if name == 'iloc':
        for i, loc in enumerate(ilocs):
            val = value.iloc[:, i]
            self._setitem_single_column(loc, val, pi)
    elif not unique_cols and value.columns.equals(self.obj.columns):
        for loc in ilocs:
            item = self.obj.columns[loc]
            if item in value:
                sub_indexer[1] = item
                val = self._align_series(tuple(sub_indexer), value.iloc[:, loc], multiindex_indexer)
            else:
                val = np.nan
            self._setitem_single_column(loc, val, pi)
    elif not unique_cols:
        raise ValueError('Setting with non-unique columns is not allowed.')
    else:
        for loc in ilocs:
            item = self.obj.columns[loc]
            if item in value:
                sub_indexer[1] = item
                val = self._align_series(tuple(sub_indexer), value[item], multiindex_indexer, using_cow=using_copy_on_write())
            else:
                val = np.nan
            self._setitem_single_column(loc, val, pi)