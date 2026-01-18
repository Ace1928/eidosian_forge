import copy
import random
import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import (
def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0):
    spec_list = []
    for i in range(len(sharding_dims)):
        random.Random(seed + i).shuffle(PLACEMENTS)
        spec_list.append(ChunkShardingSpec(dim=sharding_dims[i], placements=copy.deepcopy(PLACEMENTS)))
    return spec_list