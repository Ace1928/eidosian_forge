import inspect
import logging
import math
import re
import types
from typing import Dict, List
from torch._streambase import _StreamBase
from ..guards import install_guard
import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators
from .. import config, polyfill, variables
from ..allowed_functions import torch_get_name
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..utils import (
from .base import VariableTracker
from .ctx_manager import (
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function
def _call_ntuple(self, tx, args, kwargs):
    """inline behavior of torch.nn.modules.utils._ntuple"""
    if self.value is torch.nn.modules.utils._ntuple:
        count = args[0].as_python_constant()
    else:
        count = self.value.__closure__[0].cell_contents
    assert isinstance(count, int)
    assert not kwargs

    def handle_ntuple(value):
        if value.has_unpack_var_sequence(tx):
            return variables.TupleVariable(list(value.unpack_var_sequence(tx)))
        elif value.is_python_constant():
            return variables.ConstantVariable.create(torch.nn.modules.utils._ntuple(count)(value.as_python_constant()))
        else:
            unimplemented(f'torch.nn.modules.utils._ntuple({value})')
    if self.value is torch.nn.modules.utils._ntuple:
        return variables.LambdaVariable(handle_ntuple)
    else:
        return handle_ntuple(args[0])