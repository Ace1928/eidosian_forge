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
class SelectiveCheckpointWrapper(ActivationWrapper):

    def __init__(self, mod, memory_budget=None, policy_fn=None):
        if torch.__version__ < (2, 1):
            raise RuntimeError('SelectiveCheckpointWrapper only supported for torch >- 2.1')
        super().__init__(mod)
        if not (memory_budget is None) ^ (policy_fn is None):
            raise ValueError('Need to specify either policy_fn or memory_budget')
        self.memory_budget = memory_budget
        self.policy_fn = policy_fn
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True

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

    def get_policy_fn(self, *args, **kwargs):
        if self.policy_fn is None:
            self.policy_fn = self._get_policy_fn(*args, **kwargs)
        return self.policy_fn

    def forward(self, *args, **kwargs):
        policy_fn = self.get_policy_fn(*args, **kwargs)
        return checkpoint(self._checkpoint_wrapped_module, *args, **kwargs, policy_fn=policy_fn)