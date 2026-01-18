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
def _setitem_single_block(self, indexer, value, name: str) -> None:
    """
        _setitem_with_indexer for the case when we have a single Block.
        """
    from pandas import Series
    if isinstance(value, ABCSeries) and name != 'iloc' or isinstance(value, dict):
        value = self._align_series(indexer, Series(value))
    info_axis = self.obj._info_axis_number
    item_labels = self.obj._get_axis(info_axis)
    if isinstance(indexer, tuple):
        if self.ndim == len(indexer) == 2 and is_integer(indexer[1]) and com.is_null_slice(indexer[0]):
            col = item_labels[indexer[info_axis]]
            if len(item_labels.get_indexer_for([col])) == 1:
                loc = item_labels.get_loc(col)
                self._setitem_single_column(loc, value, indexer[0])
                return
        indexer = maybe_convert_ix(*indexer)
    if isinstance(value, ABCDataFrame) and name != 'iloc':
        value = self._align_frame(indexer, value)._values
    self.obj._check_is_chained_assignment_possible()
    self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)
    self.obj._maybe_update_cacher(clear=True, inplace=True)