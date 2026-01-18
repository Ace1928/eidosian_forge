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
def _setitem_with_indexer_missing(self, indexer, value):
    """
        Insert new row(s) or column(s) into the Series or DataFrame.
        """
    from pandas import Series
    if self.ndim == 1:
        index = self.obj.index
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The behavior of Index.insert with object-dtype is deprecated', category=FutureWarning)
            new_index = index.insert(len(index), indexer)
        if index.is_unique:
            new_indexer = index.get_indexer(new_index[-1:])
            if (new_indexer != -1).any():
                return self._setitem_with_indexer(new_indexer, value, 'loc')
        if not is_scalar(value):
            new_dtype = None
        elif is_valid_na_for_dtype(value, self.obj.dtype):
            if not is_object_dtype(self.obj.dtype):
                value = na_value_for_dtype(self.obj.dtype, compat=False)
            new_dtype = maybe_promote(self.obj.dtype, value)[0]
        elif isna(value):
            new_dtype = None
        elif not self.obj.empty and (not is_object_dtype(self.obj.dtype)):
            curr_dtype = self.obj.dtype
            curr_dtype = getattr(curr_dtype, 'numpy_dtype', curr_dtype)
            new_dtype = maybe_promote(curr_dtype, value)[0]
        else:
            new_dtype = None
        new_values = Series([value], dtype=new_dtype)._values
        if len(self.obj._values):
            new_values = concat_compat([self.obj._values, new_values])
        self.obj._mgr = self.obj._constructor(new_values, index=new_index, name=self.obj.name)._mgr
        self.obj._maybe_update_cacher(clear=True)
    elif self.ndim == 2:
        if not len(self.obj.columns):
            raise ValueError('cannot set a frame with no defined columns')
        has_dtype = hasattr(value, 'dtype')
        if isinstance(value, ABCSeries):
            value = value.reindex(index=self.obj.columns, copy=True)
            value.name = indexer
        elif isinstance(value, dict):
            value = Series(value, index=self.obj.columns, name=indexer, dtype=object)
        else:
            if is_list_like_indexer(value):
                if len(value) != len(self.obj.columns):
                    raise ValueError('cannot set a row with mismatched columns')
            value = Series(value, index=self.obj.columns, name=indexer)
        if not len(self.obj):
            df = value.to_frame().T
            idx = self.obj.index
            if isinstance(idx, MultiIndex):
                name = idx.names
            else:
                name = idx.name
            df.index = Index([indexer], name=name)
            if not has_dtype:
                df = df.infer_objects(copy=False)
            self.obj._mgr = df._mgr
        else:
            self.obj._mgr = self.obj._append(value)._mgr
        self.obj._maybe_update_cacher(clear=True)