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
def _post_backward_use_sharded_grad_views(handle: FlatParamHandle):
    if not handle._use_orig_params:
        return
    handle._reset_is_grad_none()
    handle._use_sharded_grad_views()
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            if orig_param.grad is not None and hasattr(orig_param, '_in_backward_optimizers'):
                for optim in orig_param._in_backward_optimizers:
                    optim.step()
                optim.zero_grad(set_to_none=True)
        handle._reset_flat_param_grad_info_if_needed()
        if handle._offload_params:
            handle.flat_param._cpu_grad = None