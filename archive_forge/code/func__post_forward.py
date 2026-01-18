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
def _post_forward(state: _FSDPState, handle: Optional[FlatParamHandle], reshard_fn: Callable, module: nn.Module, input: Any, output: Any) -> Any:
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    with torch.profiler.record_function('FullyShardedDataParallel._post_forward'):
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return output
        state._exec_order_data.record_post_forward(handle)
        if reshard_fn is not None:
            reshard_fn(state, handle)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output