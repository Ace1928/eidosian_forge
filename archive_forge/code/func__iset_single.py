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
def _iset_single(self, loc: int, value: ArrayLike, inplace: bool, blkno: int, blk: Block, refs: BlockValuesRefs | None=None) -> None:
    """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
    if inplace and blk.should_store(value):
        copy = False
        if using_copy_on_write() and (not self._has_no_reference_block(blkno)):
            copy = True
        iloc = self.blklocs[loc]
        blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
        return
    nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
    old_blocks = self.blocks
    new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1:]
    self.blocks = new_blocks
    return