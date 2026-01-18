import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
def _iterate_state_dict(iter_object: Any, sharded_tensor_func: Callable, dtensor_func: Callable, *, pg: Optional[dist.ProcessGroup]=None, device: Optional[torch.device]=None, cpu_offload: bool=False, ranks_only: Tuple[int, ...]=tuple()) -> Dict[str, Any]:
    cpu_device = torch.device('cpu')
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device)
    elif isinstance(iter_object, (torch.Tensor, int, float, str)) or iter_object is None:
        ret = iter_object
    elif isinstance(iter_object, dict):
        ret = {key: _iterate_state_dict(value, sharded_tensor_func, dtensor_func, pg=pg, device=device, cpu_offload=cpu_offload, ranks_only=ranks_only) for key, value in iter_object.items()}
    elif isinstance(iter_object, (list, tuple)):
        ret = [_iterate_state_dict(v, sharded_tensor_func, dtensor_func, pg=pg, device=device, cpu_offload=cpu_offload, ranks_only=ranks_only) for v in iter_object]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    else:
        raise ValueError(f'Unexpected value type {type(iter_object)}')
    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor) and cpu_offload:
            ret = ret.to(cpu_device)
    else:
        ret = {} if isinstance(ret, dict) else None
    return ret