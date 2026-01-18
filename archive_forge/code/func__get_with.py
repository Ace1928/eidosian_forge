from __future__ import annotations
from collections.abc import (
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import (
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
import pandas.plotting
def _get_with(self, key):
    if isinstance(key, ABCDataFrame):
        raise TypeError('Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column')
    elif isinstance(key, tuple):
        return self._get_values_tuple(key)
    elif not is_list_like(key):
        return self.loc[key]
    if not isinstance(key, (list, np.ndarray, ExtensionArray, Series, Index)):
        key = list(key)
    key_type = lib.infer_dtype(key, skipna=False)
    if key_type == 'integer':
        if not self.index._should_fallback_to_positional:
            return self.loc[key]
        else:
            warnings.warn('Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`', FutureWarning, stacklevel=find_stack_level())
            return self.iloc[key]
    return self.loc[key]