import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from .linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F
def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ['cpu', 'cuda', torch.utils.backend_registration._privateuse1_backend_name]
    return True