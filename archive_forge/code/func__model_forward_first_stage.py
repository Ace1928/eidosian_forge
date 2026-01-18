from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
def _model_forward_first_stage(self, tensor: TensorOrTensors, event: Event) -> None:
    try:
        assert self.model.group
        set_device_based_on_group(self.model.group)
        self.model(tensor, event=event)
    except Exception as e:
        print(f'_model_forward got {e}')
        raise e