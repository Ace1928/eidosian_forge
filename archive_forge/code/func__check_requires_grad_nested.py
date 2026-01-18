import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _check_requires_grad_nested(params: SDPAParams, debug=False) -> bool:
    if params.query.requires_grad or params.key.requires_grad or params.value.requires_grad:
        if debug:
            log.warning("Memory efficient attention currently doesn't support training with NT inputs.")
        return False
    return True