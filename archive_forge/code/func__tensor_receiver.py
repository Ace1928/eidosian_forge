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
def _tensor_receiver(cls, tensor_index):
    global _thread_local_tensor_tables
    return _thread_local_tensor_tables.recv_tables[tensor_index]