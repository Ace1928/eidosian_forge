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
def _maybe_mask_setitem_value(self, indexer, value):
    """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """
    if isinstance(indexer, tuple) and len(indexer) == 2 and isinstance(value, (ABCSeries, ABCDataFrame)):
        pi, icols = indexer
        ndim = value.ndim
        if com.is_bool_indexer(pi) and len(value) == len(pi):
            newkey = pi.nonzero()[0]
            if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                value = self.obj.iloc._align_series(indexer, value)
                indexer = (newkey, icols)
            elif isinstance(icols, np.ndarray) and icols.dtype.kind == 'i' and (len(icols) == 1):
                if ndim == 1:
                    value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)
                elif ndim == 2 and value.shape[1] == 1:
                    value = self.obj.iloc._align_frame(indexer, value)
                    indexer = (newkey, icols)
    elif com.is_bool_indexer(indexer):
        indexer = indexer.nonzero()[0]
    return (indexer, value)