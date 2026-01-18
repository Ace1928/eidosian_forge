from typing import List, Tuple
from torch.distributed.checkpoint.metadata import (
def _shards_get_overlap_region_wrt_saved_tensor(saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata) -> List[Tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.

    There returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    narrows = []
    for dim, (saved_shard_offset, current_shard_offset, saved_shard_size, current_shard_size) in enumerate(zip(saved_shard.offsets, current_shard.offsets, saved_shard.sizes, current_shard.sizes)):
        min_range_end = min(saved_shard_offset + saved_shard_size, current_shard_offset + current_shard_size)
        length = min_range_end - max(current_shard_offset, saved_shard_offset)
        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0
        narrows.append((dim, offset_for_saved_tensor, offset_for_current_tensor, length))
    return narrows