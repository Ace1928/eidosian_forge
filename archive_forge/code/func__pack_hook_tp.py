from functools import partial
from typing import Any, Optional, Tuple
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:
    """Hook function called after FWD to shard input."""
    if isinstance(x, DTensor) and all((p.is_replicate() for p in x._spec.placements)):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif not isinstance(x, DTensor) and isinstance(x, torch.Tensor) and (x.numel() >= mesh.size()):
        return DTensor.from_local(x, device_mesh=mesh).redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)]).to_local()
    else:
        return x