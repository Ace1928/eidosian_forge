from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def _sanitize_column(self, value) -> tuple[ArrayLike, BlockValuesRefs | None]:
    """
        Ensures new columns (which go into the BlockManager as new blocks) are
        always copied (or a reference is being tracked to them under CoW)
        and converted into an array.

        Parameters
        ----------
        value : scalar, Series, or array-like

        Returns
        -------
        tuple of numpy.ndarray or ExtensionArray and optional BlockValuesRefs
        """
    self._ensure_valid_index(value)
    assert not isinstance(value, DataFrame)
    if is_dict_like(value):
        if not isinstance(value, Series):
            value = Series(value)
        return _reindex_for_setitem(value, self.index)
    if is_list_like(value):
        com.require_length_match(value, self.index)
    arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
    if isinstance(value, Index) and value.dtype == 'object' and (arr.dtype != value.dtype):
        warnings.warn('Setting an Index with object dtype into a DataFrame will stop inferring another dtype in a future version. Cast the Index explicitly before setting it into the DataFrame.', FutureWarning, stacklevel=find_stack_level())
    return (arr, None)