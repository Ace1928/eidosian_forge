from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _null_wrap_accumulate_block(ignore_nulls: bool, accum_block: Callable[[AggType, Block], AggType], null_merge: Callable[[WrappedAggType, WrappedAggType], WrappedAggType]) -> Callable[[WrappedAggType, Block], WrappedAggType]:
    """
    Wrap vectorized aggregate function with null handling.

    This performs a block accumulation subject to the following null rules:
    1. If any row is null and ignore_nulls=False, return None.
    2. If at least one row is not null and ignore_nulls=True, return the block
       accumulation.
    3. If all rows are null and ignore_nulls=True, return the base accumulation.
    4. If all rows non-null, return the block accumulation.

    Args:
        ignore_nulls: Whether nulls should be ignored or cause a None result.
        accum_block: The core vectorized aggregate function to wrap.
        null_merge: A null-handling merge, as returned from _null_wrap_merge().

    Returns:
        A new vectorized aggregate function that handles nulls.
    """

    def _accum_block_null(a: WrappedAggType, block: Block) -> WrappedAggType:
        ret = accum_block(block)
        if ret is not None:
            ret = _wrap_acc(ret, has_data=True)
        elif ignore_nulls:
            ret = a
        return null_merge(a, ret)
    return _accum_block_null