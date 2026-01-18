from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def _only_sm80(op: SwiGLUOpDispatch) -> bool:
    device_type = op.device if isinstance(op.device, str) else op.device.type
    return device_type == 'cuda' and torch.cuda.get_device_capability(op.device)[0] >= 8