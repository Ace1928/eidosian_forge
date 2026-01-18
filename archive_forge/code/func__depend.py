from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast, Sequence
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers
def _depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from_idx = fork_from.find_tensor_idx()
    join_to_idx = join_to.find_tensor_idx()
    fork_from[fork_from_idx], phony = fork(fork_from[fork_from_idx])
    join_to[join_to_idx] = join(join_to[join_to_idx], phony)