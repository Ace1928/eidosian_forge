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
@torch.compiler.disable
def _get_policy_fn(self, *args, **kwargs):
    if not torch.is_grad_enabled():
        return []
    with torch.random.fork_rng():
        policy_fn = get_optimal_checkpoint_policy(self._checkpoint_wrapped_module, *args, **kwargs, memory_budget=self.memory_budget)
    if torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1):
        objects = [policy_fn]
        torch.distributed.broadcast_object_list(objects, src=0)
        policy_fn = objects[0]
    return policy_fn