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
def _new_script_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    optim = _ScriptLocalOptimizer(optim_cls, local_params_rref, *args, **kwargs)
    with _ScriptLocalOptimizer.compile_lock:
        script_optim = jit.script(optim)
        return rpc.RRef(script_optim, _ScriptLocalOptimizerInterface)