imports. For brevity, we may import the file as ``traversal_utils``.
import collections
from typing import Deque, List, Set, Tuple
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state
def _get_fsdp_handles(module: nn.Module) -> List:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_fsdp_state`.
    """
    handles = [fsdp_state._handle for fsdp_state in _get_fsdp_states(module) if fsdp_state._handle is not None]
    return handles