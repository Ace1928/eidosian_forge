import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def _unscale_grads_(self, optimizer: torch.optim.Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool=True) -> Dict[torch.device, torch.Tensor]:
    per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
    per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if not allow_fp16 and param.grad.dtype == torch.float16:
                    raise ValueError('Attempting to unscale FP16 gradients.')
                if param.grad.is_sparse:
                    if param.grad.dtype is torch.float16:
                        param_grad_fp32 = param.grad.type(torch.float32).coalesce()
                        param.grad = param_grad_fp32.type(torch.float16)
                    to_unscale = param.grad._values()
                else:
                    to_unscale = param.grad
                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)
        for device, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                if grads[0].device.type == 'cpu':
                    self._foreach_non_finite_check_and_unscale_cpu_(grads, per_device_found_inf.get(device), per_device_inv_scale.get(device))
                else:
                    torch._amp_foreach_non_finite_check_and_unscale_(grads, per_device_found_inf.get(device), per_device_inv_scale.get(device))
    if not per_device_found_inf._per_device_tensors:
        assert self._scale is not None
        per_device_found_inf.get(self._scale.device)
    return per_device_found_inf._per_device_tensors