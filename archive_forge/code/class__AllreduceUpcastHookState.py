import torch
import torch.distributed as dist
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Any, no_type_check
from torch.distributed.utils import _free_storage
@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.
    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """
    ddp_weakref: Any
    upcast_stream: torch.cuda.Stream
    wait_for_stream_enqueued: bool = False