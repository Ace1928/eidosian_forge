import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _get_ortvalues_from_torch_tensors(tensors: Tuple[torch.Tensor, ...], devices: Tuple['ORTC.OrtDevice', ...]) -> Tuple[torch.Tensor, ...]:
    ortvalues = ORTC.OrtValueVector()
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []
    for tensor in tensors:
        dtypes.append(_TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues