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
def _py_rref_reducer(self, py_rref):
    rref_fork_data = py_rref._serialize()
    return (_InternalRPCPickler._py_rref_receiver, (rref_fork_data,))