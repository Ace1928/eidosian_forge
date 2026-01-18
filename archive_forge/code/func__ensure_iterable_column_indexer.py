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
def _ensure_iterable_column_indexer(self, column_indexer):
    """
        Ensure that our column indexer is something that can be iterated over.
        """
    ilocs: Sequence[int | np.integer] | np.ndarray
    if is_integer(column_indexer):
        ilocs = [column_indexer]
    elif isinstance(column_indexer, slice):
        ilocs = np.arange(len(self.obj.columns))[column_indexer]
    elif isinstance(column_indexer, np.ndarray) and column_indexer.dtype.kind == 'b':
        ilocs = np.arange(len(column_indexer))[column_indexer]
    else:
        ilocs = column_indexer
    return ilocs