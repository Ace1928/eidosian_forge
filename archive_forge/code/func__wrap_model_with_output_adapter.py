from __future__ import annotations
import contextlib
import functools
import inspect
from typing import (
import torch._dynamo
import torch.export as torch_export
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree
def _wrap_model_with_output_adapter(model: Union[torch.nn.Module, Callable], output_adapter: DynamoFlattenOutputStep) -> Callable:
    """Wrap model with output adapter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output adapter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The adapting logic is controlled by ``output_adapter``.

    Args:
        model: PyTorch model or function.
        output_adapter: Output adapter to apply to model output.
    Returns:
        Wrapped model.
    """
    model_func = model.forward if isinstance(model, torch.nn.Module) else model

    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        return output_adapter.apply(model_func(*args, **kwargs), model=model)
    return wrapped