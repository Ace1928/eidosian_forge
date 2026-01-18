from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
def _wrap_for_symbolic_trace(target: Callable) -> Tuple[Callable, Callable]:
    """This function wraps ```target`` for symbolic tracing.

    This function wraps ```target``` so that its wrapper produces
    torch.fx.Proxy in symbolic computation. The returned values are
    the wrapper and then the original function. Per `_TORCH_METHODS_TO_PATCH`,
    this function shall receive `torch.arange`, `torch.tensor`, etc. as inputs.
    """

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v
        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)
        if proxy is not None:
            return proxy.tracer.create_proxy('call_function', target, args, kwargs)
        else:
            return target(*args, **kwargs)
    return (wrapper, target)