from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias
def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    if device.type not in set(_get_foreach_kernels_supported_devices() + ['cpu']) or torch.jit.is_scripting():
        return False
    return all((t is None or type(t) == torch.Tensor for t in tensors))