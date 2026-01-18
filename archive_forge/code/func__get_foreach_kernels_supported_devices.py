from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias
def _get_foreach_kernels_supported_devices() -> List[str]:
    """Return the device type list that supports foreach kernels."""
    return ['cuda', 'xpu', torch._C._get_privateuse1_backend_name()]