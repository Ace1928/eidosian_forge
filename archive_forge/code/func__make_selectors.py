from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def _make_selectors(self):
    new_levels = self.new_index_levels
    remaining_labels = self.sorted_labels[:-1]
    level_sizes = tuple((len(x) for x in new_levels))
    comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
    ngroups = len(obs_ids)
    comp_index = ensure_platform_int(comp_index)
    stride = self.index.levshape[self.level] + self.lift
    self.full_shape = (ngroups, stride)
    selector = self.sorted_labels[-1] + stride * comp_index + self.lift
    mask = np.zeros(np.prod(self.full_shape), dtype=bool)
    mask.put(selector, True)
    if mask.sum() < len(self.index):
        raise ValueError('Index contains duplicate entries, cannot reshape')
    self.group_index = comp_index
    self.mask = mask
    if self.sort:
        self.compressor = comp_index.searchsorted(np.arange(ngroups))
    else:
        self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])