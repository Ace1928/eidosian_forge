import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
def _call_min_max_binary(self, tx, a, b):
    if self.tensor_args(a, b):
        if not isinstance(a, variables.TensorVariable):
            a, b = (b, a)
        assert isinstance(a, variables.TensorVariable)
        if isinstance(a, FakeItemVariable):
            a = variables.TorchVariable(torch.tensor).call_function(tx, [a], {})
        if isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            from .builder import wrap_fx_proxy_cls
            return wrap_fx_proxy_cls(type(a), tx=tx, proxy=tx.output.create_proxy('call_function', self.fn, *proxy_args_kwargs([a, b], {})))
        if b.is_python_constant():
            if isinstance(a, variables.NumpyNdarrayVariable):
                import numpy as np
                fn = variables.NumpyVariable(np.clip)
            else:
                fn = variables.TorchVariable(torch.clamp)
            kwargs = {'min': b} if self.fn is max else {'max': b}
            result = fn.call_function(tx, [a], kwargs)
        else:
            if isinstance(a, variables.NumpyNdarrayVariable):
                import numpy as np
                fn = {max: np.maximum, min: np.minimum}[self.fn]
                fn = variables.NumpyVariable(fn)
            else:
                fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                fn = variables.TorchVariable(fn)
            result = fn.call_function(tx, [a, b], {})
        if all((isinstance(i, (variables.UnspecializedPythonVariable, variables.ConstantVariable)) for i in [a, b])):
            if any((isinstance(val, FakeItemVariable) for val in [a, b])):
                return variables.FakeItemVariable.from_tensor_variable(result)
            if b.is_python_constant():
                raw_b = b.as_python_constant()
            else:
                raw_b = b.raw_value
            if self.fn is max:
                raw_res = max(a.raw_value, raw_b)
            else:
                raw_res = min(a.raw_value, raw_b)
            need_unwrap = any((x.need_unwrap for x in [a, b] if isinstance(x, variables.UnspecializedPythonVariable)))
            return variables.UnspecializedPythonVariable.from_tensor_variable(result, raw_res, need_unwrap)
        else:
            return result
    elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
        proxy = tx.output.create_proxy('call_function', self.fn, *proxy_args_kwargs([a, b], {}))
        return SymNodeVariable.create(tx, proxy, None)