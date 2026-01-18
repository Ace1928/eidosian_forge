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
def _register_post_backward_reshard_only_hook(state: _FSDPState, handle: Optional[FlatParamHandle], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """
    Registers post-backward hooks to reshard flat parameters that do not
    require gradient. We register these using multi-post-grad hooks on the
    input activations to ensure that all gradients that may depend on the
    parameters have been computed before resharding.
    """
    if not torch.is_grad_enabled():
        return
    inp_tensors: Optional[List[torch.Tensor]] = None
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, '_post_backward_hook_state')
    if already_registered or flat_param.requires_grad:
        return
    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad]
    assert inp_tensors is not None
    hook_handle = register_multi_grad_hook(inp_tensors, functools.partial(_post_backward_reshard, state, handle))
    flat_param._post_backward_hook_state = (hook_handle,)