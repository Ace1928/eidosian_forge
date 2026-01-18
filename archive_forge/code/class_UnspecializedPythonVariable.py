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
class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, *, raw_value=None, need_unwrap=True, **kwargs):
        super().__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value, need_unwrap=True):
        return UnspecializedPythonVariable(**dict(tensor_variable.__dict__), raw_value=raw_value, need_unwrap=need_unwrap)