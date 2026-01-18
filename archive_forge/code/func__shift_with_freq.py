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
def _shift_with_freq(self, periods: int, axis: int, freq) -> Self:
    index = self._get_axis(axis)
    if freq == 'infer':
        freq = getattr(index, 'freq', None)
        if freq is None:
            freq = getattr(index, 'inferred_freq', None)
        if freq is None:
            msg = 'Freq was not set in the index hence cannot be inferred'
            raise ValueError(msg)
    elif isinstance(freq, str):
        is_period = isinstance(index, PeriodIndex)
        freq = to_offset(freq, is_period=is_period)
    if isinstance(index, PeriodIndex):
        orig_freq = to_offset(index.freq)
        if freq != orig_freq:
            assert orig_freq is not None
            raise ValueError(f'Given freq {freq_to_period_freqstr(freq.n, freq.name)} does not match PeriodIndex freq {freq_to_period_freqstr(orig_freq.n, orig_freq.name)}')
        new_ax = index.shift(periods)
    else:
        new_ax = index.shift(periods, freq)
    result = self.set_axis(new_ax, axis=axis)
    return result.__finalize__(self, method='shift')