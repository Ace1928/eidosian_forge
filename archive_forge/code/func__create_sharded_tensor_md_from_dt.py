import copy
from typing import Any, cast, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor.parallel._data_parallel_utils import (
def _create_sharded_tensor_md_from_dt(dt: DTensor, dt_pg: c10d.ProcessGroup) -> ShardedTensorMetadata:
    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1
    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1
    for i in range(shard_count):
        offsets, sizes = _get_box_for(dt, i)
        shards_md.append(ShardMetadata(shard_offsets=list(offsets), shard_sizes=list(sizes), placement=f'rank:{(scapegoat_rank if i > 0 else my_rank)}/{dt._local_tensor.device}'))
    return ShardedTensorMetadata(shards_metadata=shards_md, size=dt.size(), tensor_properties=TensorProperties(dtype=dt.dtype, layout=dt.layout, requires_grad=dt.requires_grad))