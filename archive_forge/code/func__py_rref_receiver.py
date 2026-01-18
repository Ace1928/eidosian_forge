import collections
import copyreg
import io
import pickle
import sys
import threading
import traceback
from enum import Enum
import torch
import torch.distributed as dist
from torch._C._distributed_rpc import _get_current_rpc_agent
@classmethod
def _py_rref_receiver(cls, rref_fork_data):
    return dist.rpc.PyRRef._deserialize(rref_fork_data)