from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _combine(self, blocks: list[Block], index: Index | None=None) -> Self:
    """return a new manager with the blocks"""
    if len(blocks) == 0:
        if self.ndim == 2:
            if index is not None:
                axes = [self.items[:0], index]
            else:
                axes = [self.items[:0]] + self.axes[1:]
            return self.make_empty(axes)
        return self.make_empty()
    indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
    inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])
    new_blocks: list[Block] = []
    for b in blocks:
        nb = b.copy(deep=False)
        nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
        new_blocks.append(nb)
    axes = list(self.axes)
    if index is not None:
        axes[-1] = index
    axes[0] = self.items.take(indexer)
    return type(self).from_blocks(new_blocks, axes)