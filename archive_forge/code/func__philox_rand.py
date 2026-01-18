from typing import Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.types import _device, _dtype
def _philox_rand(shape: torch.Size, seed: torch.Tensor, offset: torch.Tensor, stride: Optional[Tuple[int, ...]], device: _device, dtype: _dtype):
    assert stride is None
    if device.type == 'cpu':
        devices = []
    else:
        devices = [device]
    if device.type != 'cuda':
        raise throw_on_non_cuda(device)
    with torch.random.fork_rng(devices):
        CUDARngStateHelper.set_torch_state_tensor(seed, offset)
        random_values = torch.rand(shape, device=device, dtype=dtype)
    return (random_values, philox_rand_offset(shape))