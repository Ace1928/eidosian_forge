from __future__ import annotations
from typing import (
from pandas.core.dtypes.common import is_1d_only_ea_dtype
def _iter_block_pairs(left: BlockManager, right: BlockManager) -> Iterator[BlockPairInfo]:
    for blk in left.blocks:
        locs = blk.mgr_locs
        blk_vals = blk.values
        left_ea = blk_vals.ndim == 1
        rblks = right._slice_take_blocks_ax0(locs.indexer, only_slice=True)
        for rblk in rblks:
            right_ea = rblk.values.ndim == 1
            lvals, rvals = _get_same_shape_values(blk, rblk, left_ea, right_ea)
            info = BlockPairInfo(lvals, rvals, locs, left_ea, right_ea, rblk)
            yield info