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
def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)