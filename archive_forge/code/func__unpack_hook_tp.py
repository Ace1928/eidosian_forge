from functools import partial
from typing import Any, Optional, Tuple
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:
    """Hook function called before activation recomputing in BWD to restore input."""
    if isinstance(x, DTensor) and len(x._spec.placements) == 1 and x._spec.placements[0].is_shard():
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif not isinstance(x, DTensor) and isinstance(x, torch.Tensor) and (x.numel() >= mesh.size()):
        return DTensor.from_local(x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]).redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
    else:
        return x