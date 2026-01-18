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
def _has_valid_setitem_indexer(self, indexer) -> bool:
    """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Returns
        -------
        bool
        """
    if isinstance(indexer, dict):
        raise IndexError('iloc cannot enlarge its target object')
    if isinstance(indexer, ABCDataFrame):
        raise TypeError('DataFrame indexer for .iloc is not supported. Consider using .loc with a DataFrame indexer for automatic alignment.')
    if not isinstance(indexer, tuple):
        indexer = _tuplify(self.ndim, indexer)
    for ax, i in zip(self.obj.axes, indexer):
        if isinstance(i, slice):
            pass
        elif is_list_like_indexer(i):
            pass
        elif is_integer(i):
            if i >= len(ax):
                raise IndexError('iloc cannot enlarge its target object')
        elif isinstance(i, dict):
            raise IndexError('iloc cannot enlarge its target object')
    return True