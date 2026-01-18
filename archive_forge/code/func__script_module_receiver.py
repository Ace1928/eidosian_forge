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
def _script_module_receiver(cls, script_module_serialized):
    """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
    f = io.BytesIO(script_module_serialized)
    m = torch.jit.load(f)
    return m