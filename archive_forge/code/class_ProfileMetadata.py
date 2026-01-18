import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
@dataclass
class ProfileMetadata:
    name: str
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: Any
    inplace_info: Tuple[int, int]
    is_view_like: bool
    is_rand_op: bool