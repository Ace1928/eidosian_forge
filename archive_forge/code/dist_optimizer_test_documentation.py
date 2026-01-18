import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (

    Call rpc.rpc_async on a method in a remote object.

    Args:
        method: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method

    Returns a Future to the method call result.
    