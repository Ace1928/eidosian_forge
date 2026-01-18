from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.cuda.Stream
    wait_for_optim_stream_enqueued: bool