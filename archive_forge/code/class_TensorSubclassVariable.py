import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
class TensorSubclassVariable(VariableTracker):

    def __init__(self, value, *args, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        if len(args) == 1 and isinstance(args[0], TensorVariable):
            from .builder import VariableBuilder
            from .torch_function import TensorWithTFOverrideVariable
            torch_fn = VariableBuilder(tx, AttrSource(self.source, '__torch_function__'))(self.value.__torch_function__)
            return TensorWithTFOverrideVariable.from_tensor_var(tx, args[0], self.value, torch_fn)
        return super().call_function(tx, args, kwargs)