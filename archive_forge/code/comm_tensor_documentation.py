from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only

    A Tensor subclass to wrap input tensors for collective communications.

    This Tensor subclass works for both eager and tracing mode.
    In eager mode, it will record whether the inplace collective communication
    has been launched using this Tensor and remember the corresponding work
    handle. If yes, it will explicitly call wait() in the ``__torch_dispatch__``
    function before subsequent operations consuming the value of the Tensor.

    In tracing mode, ``CommTensor`` inserts two node into the graph using the
    ``__torch_dispatch__`` function.
    1. The first node is inserted right after the
    communication, wrapping both the inplace output tensor and the returned
    work handle into a custom ``_CommResult`` type. We have to do this because
    ``ProxyTorchDispatchMode`` only handles ``torch.Tensor``, ``_ProxyTensor``,
    and ``torch.nn.Parameter`` objects and will treat the work handle
    as a constant and embed that into the graph. As a result, during execution,
    it will use the work handle created during tracing and will lead to wrong
    result. The solution in this test is to manually create a proxy on the
    return value of ``allreduce_`` which is ``([tensor], work)``, and wrap that
    to ``[(_CommResult(tensor, work)), work]``. In this way, subsequent nodes can
    directly consume ``_CommResult``.
    2. The second node is inserted right before any subsequent node reads from
    ``_CommResult``. It will call ``wait()`` on the stashed work handle to ensure
    that computation waits for communication.
    