from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
def _foreach_worker(self, callback: Callable, args: Any=None) -> None:
    futures = [rpc.rpc_async(self._get_rpc_name(rank), callback, args=args) for rank in range(1, self.group.size())]
    futures = [f.wait() for f in futures]