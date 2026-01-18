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
class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """

    def __init__(self, device_handle) -> None:
        super().__init__()
        self.compute_stream = None
        self.device_handle = device_handle
        self.post_unflatten_transform = torch._dynamo.disable(self.post_unflatten_transform)

    def pre_flatten_transform(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[Any]]:
        return _flatten_tensor(tensor)

    def post_unflatten_transform(self, tensor: torch.Tensor, param_extension: Any) -> torch.Tensor:
        stream = self.compute_stream or self.device_handle.current_stream()
        with self.device_handle.stream(stream):
            result = _unflatten_tensor(tensor, param_extension, device_handle=self.device_handle, compute_stream=self.compute_stream)
            _set_fsdp_flattened(result)
            return result

    def chunk_tensor(self, tensor: torch.Tensor, rank: int, world_size: int, num_devices_per_node: int, pg: dist.ProcessGroup, device: Optional[torch.device]=None) -> torch.Tensor:
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
        return _chunk_dtensor(tensor, rank, device_mesh)

    def pre_load_state_dict_transform(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[Shard]]:
        return _pre_load_state_dict(tensor)

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: Optional[DeviceMesh]) -> torch.Tensor:
        return _all_gather_dtensor(tensor, parent_mesh)