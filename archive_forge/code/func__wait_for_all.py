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
def _wait_for_all(rpc_futs):
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results