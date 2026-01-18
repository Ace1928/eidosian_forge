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
def _flex_arith_method(self, other, op, *, axis: Axis='columns', level=None, fill_value=None):
    axis = self._get_axis_number(axis) if axis is not None else 1
    if self._should_reindex_frame_op(other, op, axis, fill_value, level):
        return self._arith_method_with_reindex(other, op)
    if isinstance(other, Series) and fill_value is not None:
        raise NotImplementedError(f'fill_value {fill_value} not supported.')
    other = ops.maybe_prepare_scalar_for_op(other, self.shape)
    self, other = self._align_for_op(other, axis, flex=True, level=level)
    with np.errstate(all='ignore'):
        if isinstance(other, DataFrame):
            new_data = self._combine_frame(other, op, fill_value)
        elif isinstance(other, Series):
            new_data = self._dispatch_frame_op(other, op, axis=axis)
        else:
            if fill_value is not None:
                self = self.fillna(fill_value)
            new_data = self._dispatch_frame_op(other, op)
    return self._construct_result(new_data)