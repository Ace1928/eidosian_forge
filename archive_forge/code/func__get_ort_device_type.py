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
def _get_ort_device_type(device_type: str):
    if device_type == 'cuda':
        return ORTC.OrtDevice.cuda()
    if device_type == 'cpu':
        return ORTC.OrtDevice.cpu()
    if device_type == 'ort':
        return ORTC.OrtDevice.npu()
    raise ValueError('Unsupported device type: ' + device_type)