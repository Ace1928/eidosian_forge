from typing import List, Tuple
from ray.data._internal.block_list import BlockList
from ray.data._internal.split import _calculate_blocks_rows, _split_at_indices
from ray.data.block import Block, BlockMetadata, BlockPartition
from ray.types import ObjectRef
def _shave_one_split(split: BlockPartition, num_rows_per_block: List[int], target_size: int) -> Tuple[BlockPartition, int, BlockPartition]:
    """Shave a block list to the target size.

    Args:
        split: the block list to shave.
        num_rows_per_block: num rows for each block in the list.
        target_size: the upper bound target size of the shaved list.
    Returns:
        A tuple of:
            - shaved block list.
            - num of rows needed for the block list to meet the target size.
            - leftover blocks.

    """
    shaved = []
    leftovers = []
    shaved_rows = 0
    for block_with_meta, block_rows in zip(split, num_rows_per_block):
        if block_rows + shaved_rows <= target_size:
            shaved.append(block_with_meta)
            shaved_rows += block_rows
        else:
            leftovers.append(block_with_meta)
    num_rows_needed = target_size - shaved_rows
    return (shaved, num_rows_needed, leftovers)