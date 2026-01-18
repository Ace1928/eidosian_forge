import logging
from collections import defaultdict
from threading import Lock
from typing import List, Optional
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.distributed.rpc import RRef
from .utils import functional_optim_map
class _ScriptLocalOptimizer(nn.Module):
    compile_lock = Lock()

    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        super().__init__()
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    @jit.export
    def step(self, autograd_ctx_id: int):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
        grads: List[Optional[Tensor]] = [all_local_grads[p] if p in all_local_grads else None for p in self._local_params]
        self.optim.step(grads)