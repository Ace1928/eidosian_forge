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
def _reindex_with_indexers(self, reindexers, fill_value=None, copy: bool_t | None=False, allow_dups: bool_t=False) -> Self:
    """allow_dups indicates an internal call here"""
    new_data = self._mgr
    for axis in sorted(reindexers.keys()):
        index, indexer = reindexers[axis]
        baxis = self._get_block_manager_axis(axis)
        if index is None:
            continue
        index = ensure_index(index)
        if indexer is not None:
            indexer = ensure_platform_int(indexer)
        new_data = new_data.reindex_indexer(index, indexer, axis=baxis, fill_value=fill_value, allow_dups=allow_dups, copy=copy)
        copy = False
    if (copy or copy is None) and new_data is self._mgr and (not using_copy_on_write()):
        new_data = new_data.copy(deep=copy)
    elif using_copy_on_write() and new_data is self._mgr:
        new_data = new_data.copy(deep=False)
    return self._constructor_from_mgr(new_data, axes=new_data.axes).__finalize__(self)