imports. For brevity, we may import the file as ``traversal_utils``.
import collections
from typing import Deque, List, Set, Tuple
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state
def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
    registry = _get_registry(module)
    if registry is None:
        return True
    return 'replicate' not in registry