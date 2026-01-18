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
@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable = False

    def _convert_key(self, key):
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
        if self.ndim == 1 and len(key) > 1:
            key = (key,)
        return key

    @property
    def _axes_are_unique(self) -> bool:
        assert self.ndim == 2
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key):
        if self.ndim == 2 and (not self._axes_are_unique):
            if not isinstance(key, tuple) or not all((is_scalar(x) for x in key)):
                raise ValueError('Invalid call for scalar access (getting)!')
            return self.obj.loc[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if self.ndim == 2 and (not self._axes_are_unique):
            if not isinstance(key, tuple) or not all((is_scalar(x) for x in key)):
                raise ValueError('Invalid call for scalar access (setting)!')
            self.obj.loc[key] = value
            return
        return super().__setitem__(key, value)