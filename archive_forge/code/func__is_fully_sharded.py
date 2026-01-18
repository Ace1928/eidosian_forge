import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel
from .contract import _get_registry, contract
def _is_fully_sharded(module: nn.Module) -> bool:
    """Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return 'fully_shard' in registry