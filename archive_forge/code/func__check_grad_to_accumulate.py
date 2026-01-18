import functools
import logging
from enum import auto, Enum
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import (
from torch.utils import _pytree as pytree
def _check_grad_to_accumulate(new_sharded_grad: torch.Tensor, accumulated_grad: torch.Tensor) -> None:
    _p_assert(accumulated_grad.shape == new_sharded_grad.shape, f'Shape mismatch when accumulating gradients: existing gradient shape={accumulated_grad.shape} new gradient shape={new_sharded_grad.shape}')
    _p_assert(accumulated_grad.device == new_sharded_grad.device, f'Device mismatch when accumulating gradients: existing gradient device={accumulated_grad.device} new gradient device={new_sharded_grad.device}')