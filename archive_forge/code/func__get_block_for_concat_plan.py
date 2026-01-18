from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _get_block_for_concat_plan(mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int) -> Block:
    blk = mgr.blocks[blkno]
    if len(bp) == len(blk.mgr_locs) and (blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1):
        nb = blk
    else:
        ax0_blk_indexer = mgr.blklocs[bp.indexer]
        slc = lib.maybe_indices_to_slice(ax0_blk_indexer, max_len)
        if isinstance(slc, slice):
            nb = blk.slice_block_columns(slc)
        else:
            nb = blk.take_block_columns(slc)
    return nb