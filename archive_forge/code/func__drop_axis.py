from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def _drop_axis(self, labels, axis, level=None, errors: IgnoreRaise='raise', only_slice: bool_t=False) -> Self:
    """
        Drop labels from specified axis. Used in the ``drop`` method
        internally.

        Parameters
        ----------
        labels : single label or list-like
        axis : int or axis name
        level : int or level name, default None
            For MultiIndex
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.
        only_slice : bool, default False
            Whether indexing along columns should be view-only.

        """
    axis_num = self._get_axis_number(axis)
    axis = self._get_axis(axis)
    if axis.is_unique:
        if level is not None:
            if not isinstance(axis, MultiIndex):
                raise AssertionError('axis must be a MultiIndex')
            new_axis = axis.drop(labels, level=level, errors=errors)
        else:
            new_axis = axis.drop(labels, errors=errors)
        indexer = axis.get_indexer(new_axis)
    else:
        is_tuple_labels = is_nested_list_like(labels) or isinstance(labels, tuple)
        labels = ensure_object(common.index_labels_to_array(labels))
        if level is not None:
            if not isinstance(axis, MultiIndex):
                raise AssertionError('axis must be a MultiIndex')
            mask = ~axis.get_level_values(level).isin(labels)
            if errors == 'raise' and mask.all():
                raise KeyError(f'{labels} not found in axis')
        elif isinstance(axis, MultiIndex) and labels.dtype == 'object' and (not is_tuple_labels):
            mask = ~axis.get_level_values(0).isin(labels)
        else:
            mask = ~axis.isin(labels)
            labels_missing = (axis.get_indexer_for(labels) == -1).any()
            if errors == 'raise' and labels_missing:
                raise KeyError(f'{labels} not found in axis')
        if isinstance(mask.dtype, ExtensionDtype):
            mask = mask.to_numpy(dtype=bool)
        indexer = mask.nonzero()[0]
        new_axis = axis.take(indexer)
    bm_axis = self.ndim - axis_num - 1
    new_mgr = self._mgr.reindex_indexer(new_axis, indexer, axis=bm_axis, allow_dups=True, copy=None, only_slice=only_slice)
    result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
    if self.ndim == 1:
        result._name = self.name
    return result.__finalize__(self)