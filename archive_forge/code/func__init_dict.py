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
def _init_dict(self, data, index: Index | None=None, dtype: DtypeObj | None=None):
    """
        Derive the "_mgr" and "index" attributes of a new Series from a
        dictionary input.

        Parameters
        ----------
        data : dict or dict-like
            Data used to populate the new Series.
        index : Index or None, default None
            Index for the new Series: if None, use dict keys.
        dtype : np.dtype, ExtensionDtype, or None, default None
            The dtype for the new Series: if None, infer from data.

        Returns
        -------
        _data : BlockManager for the new Series
        index : index for the new Series
        """
    keys: Index | tuple
    if data:
        keys = tuple(data.keys())
        values = list(data.values())
    elif index is not None:
        if len(index) or dtype is not None:
            values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
        else:
            values = []
        keys = index
    else:
        keys, values = (default_index(0), [])
    s = Series(values, index=keys, dtype=dtype)
    if data and index is not None:
        s = s.reindex(index, copy=False)
    return (s._mgr, s.index)