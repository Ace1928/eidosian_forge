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
def _send_result_and_do_backwards(training: bool, message: PipeMessage, grads_message: PipeMessage) -> None:
    group = get_pipeline_parallel_group()
    set_device_based_on_group(group)
    result = PipeResult
    model = PipeModel
    if isinstance(result, torch.Tensor):
        result = tuple([result])
    message.tensors = tuple(result)
    assert model.pipeline
    transport = model.pipeline.transport
    transport.send_message(message, sync=False, skip_header=True)
    if training:
        grads_message.tensor_shapes = [r.shape for r in result]
        grads_message.tensor_dtypes = [r.dtype for r in result]
        grads_message = transport.recv_message_tensors(grads_message)
        with model.lock:
            torch.autograd.backward(result, grads_message.tensors, retain_graph=True)