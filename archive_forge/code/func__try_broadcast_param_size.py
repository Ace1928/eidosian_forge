import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _try_broadcast_param_size(q_size, k_size, v_size, param_name, debug=False) -> bool:
    max_size = max(q_size, k_size, v_size)
    if q_size != max_size and q_size != 1 or (k_size != max_size and k_size != 1) or (v_size != max_size and v_size != 1):
        if debug:
            log.warning('Both fused kernels require query, key and value to have broadcastable %s, got Query %s %d, Key %s %d, Value %s %d instead.', param_name, param_name, q_size, param_name, k_size, param_name, v_size)
        return False
    return True