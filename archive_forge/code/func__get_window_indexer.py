from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _get_window_indexer(self) -> GroupbyIndexer:
    """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
    rolling_indexer: type[BaseIndexer]
    indexer_kwargs: dict[str, Any] | None = None
    index_array = self._index_array
    if isinstance(self.window, BaseIndexer):
        rolling_indexer = type(self.window)
        indexer_kwargs = self.window.__dict__.copy()
        assert isinstance(indexer_kwargs, dict)
        indexer_kwargs.pop('index_array', None)
        window = self.window
    elif self._win_freq_i8 is not None:
        rolling_indexer = VariableWindowIndexer
        window = self._win_freq_i8
    else:
        rolling_indexer = FixedWindowIndexer
        window = self.window
    window_indexer = GroupbyIndexer(index_array=index_array, window_size=window, groupby_indices=self._grouper.indices, window_indexer=rolling_indexer, indexer_kwargs=indexer_kwargs)
    return window_indexer