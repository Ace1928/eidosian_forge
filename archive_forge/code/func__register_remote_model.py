from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
@staticmethod
def _register_remote_model(args: List[Any], kwargs: Dict[str, Any]) -> None:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    kwargs['group'] = group
    kwargs['input_device'] = torch.device('cuda', torch.cuda.current_device())
    model = AsyncPipe(*args, **kwargs)
    model.cuda()
    global PipeModel
    PipeModel = model