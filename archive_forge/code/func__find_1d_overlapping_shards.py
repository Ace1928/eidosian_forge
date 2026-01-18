from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def _find_1d_overlapping_shards(shards: List[ShardMetadata], dim: int) -> Optional[Tuple[int, int]]:
    intervals = [(s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i) for i, s in enumerate(shards)]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None