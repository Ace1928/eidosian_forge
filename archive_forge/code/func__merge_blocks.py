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
def _merge_blocks(blocks: list[Block], dtype: DtypeObj, can_consolidate: bool) -> tuple[list[Block], bool]:
    if len(blocks) == 1:
        return (blocks, False)
    if can_consolidate:
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])
        new_values: ArrayLike
        if isinstance(blocks[0].dtype, np.dtype):
            new_values = np.vstack([b.values for b in blocks])
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)
        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]
        bp = BlockPlacement(new_mgr_locs)
        return ([new_block_2d(new_values, placement=bp)], True)
    return (blocks, False)