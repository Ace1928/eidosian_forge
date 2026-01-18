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
@no_type_check
def _root_cast_forward_input(state: _FSDPState, module: torch.nn.Module, args, kwargs) -> Tuple[Any, Any]:
    if state._handle:
        force_full_precision = not state._handle._force_full_precision
    else:
        force_full_precision = True
    should_cast_forward_inputs = ((module.training or not state._use_full_prec_in_eval) and force_full_precision) and state.mixed_precision.cast_root_forward_inputs
    if should_cast_forward_inputs:
        input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
    return (args, kwargs)