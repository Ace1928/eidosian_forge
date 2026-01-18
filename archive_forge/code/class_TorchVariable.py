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
class TorchVariable(BaseTorchVariable):
    """Points to a module, classes or functions in torch.*"""

    def __init__(self, value, **kwargs):
        assert not isinstance(value, (torch.dtype, torch.device)), 'should use ConstantVariable'
        super().__init__(value, **kwargs)
        try:
            self_should_be_none = getattr(self.value, '__self__', None)
        except RuntimeError as e:
            assert 'No such operator' in str(e), str(e)
            self_should_be_none = None
        except AssertionError as e:
            assert 'Unknown attribute' in str(e), str(e)
            self_should_be_none = None
        if self_should_be_none is None:
            pass
        elif isinstance(self_should_be_none, types.ModuleType):
            name = self_should_be_none.__name__
            assert re.match('^(torch|math)([.]|$)', name), f'__self__ set to {name}'
        elif isinstance(self_should_be_none, type(torch._C._get_tracing_state.__self__)):
            pass
        elif isinstance(self_should_be_none, torch_special_class_types):
            pass
        else:
            raise AssertionError(f'{value} found with __self__ set')

    def __repr__(self):
        return f'TorchVariable({self.value})'

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module, torch.device)):
            return type(self.value)
        if isinstance(self.value, type):
            return type
        return super().python_type()

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable
        from .builder import wrap_fx_proxy
        constant_args = check_constant_args(args, kwargs)
        unspec_python_args = check_unspec_python_args(args, kwargs)
        if self.can_constant_fold_through() and (constant_args or unspec_python_args):
            return ConstantVariable.create(self.as_python_constant()(*[x.as_python_constant() for x in args], **{k: v.as_python_constant() for k, v in kwargs.items()}))
        elif istype(self.value, type) and issubclass(self.value, torch.nn.Module):
            if self.value is torch.nn.CrossEntropyLoss:
                return self._call_cross_entropy_loss(tx, args, kwargs)
            else:
                return variables.UserDefinedClassVariable(self.value, source=self.source).call_function(tx, args, kwargs)
        elif can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)
        elif isinstance(self.value, types.ModuleType):
            unimplemented('TypeError("\'module\' object is not callable")')
        else:
            if np and self.value in tensortype_to_dtype and (len(args) == 1) and isinstance(args[0], ListVariable) and (len(args[0].items) > 1) and all((isinstance(x, variables.TensorVariable) for x in args[0].items)):
                stacked = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', torch.stack, *proxy_args_kwargs(args, kwargs)))
                args = [stacked]
            tensor_variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, *proxy_args_kwargs(args, kwargs)))
            return tensor_variable

    def _call_cross_entropy_loss(self, tx, args, kwargs):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(weight=ConstantVariable.create(None), size_average=ConstantVariable.create(None), ignore_index=ConstantVariable.create(-100), reduce=ConstantVariable.create(None), reduction=ConstantVariable.create('mean'), label_smoothing=ConstantVariable.create(0.0)):
            return (weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        weight, size_average, ignore_index, reduce_arg, reduction, label_smoothing = normalize_args(*args, **kwargs)

        def fake_cross_entropy_loss(input, target):
            from .builder import wrap_fx_proxy
            return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', torch.nn.functional.cross_entropy, *proxy_args_kwargs([input, target, weight, size_average, ignore_index, reduce_arg, reduction, label_smoothing], {})))
        return variables.LambdaVariable(fake_cross_entropy_loss)