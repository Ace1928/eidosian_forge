import inspect
from typing import Dict, List
import torch.utils._pytree as pytree
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalSource
from ..utils import is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .tensor import TensorVariable
from .user_defined import UserDefinedClassVariable
class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    def __init__(self, *args, **kwargs):
        self.torch_function_fn = kwargs.pop('torch_function_fn')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_tensor_var(cls, tx, tensor_var, class_type, torch_function_fn):
        import torch
        kwargs = dict(tensor_var.__dict__)
        assert kwargs.pop('class_type') is torch.Tensor, 'invalid class type in TensorWithTFOverrideVariable.from_tensor_var'
        var = cls(torch_function_fn=torch_function_fn, class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx):
        if self.global_mangled_class_name() not in tx.output.global_scope:
            tx.output.install_global(self.global_mangled_class_name(), self.class_type)

    def python_type(self):
        return self.class_type

    def subclass_type_var(self):
        return UserDefinedClassVariable(self.class_type, source=GlobalSource(self.global_mangled_class_name()))

    def global_mangled_class_name(self):
        return f'__subclass_{self.class_type.__name__}_{id(self.class_type)}'

    def var_getattr(self, tx, name):
        import torch
        from .builder import SourcelessBuilder
        if name in banned_attrs or not hasattr(torch.Tensor, name):
            unimplemented(f'Accessing {name} on a tensor subclass with a __torch_function__ override is not supported')
        if _is_attr_overidden(tx, self, name):
            unimplemented(f'Accessing overridden method/attribute {name} on a tensor subclass with a __torch_function__ override is not supported')
        if tx.output.torch_function_enabled:
            if self.source:
                install_guard(AttrSource(AttrSource(self.source, '__class__'), name).make_guard(GuardBuilder.FUNCTION_MATCH))
            get_fn = SourcelessBuilder()(tx, getattr(torch.Tensor, name).__get__)
            return self.call_torch_function(tx, get_fn, TupleVariable([self.subclass_type_var()]), [self], {})
        else:
            return super().var_getattr(tx, name)

    def call_torch_function(self, tx, fn, types, args, kwargs):
        return call_torch_function(tx, self.subclass_type_var(), self.torch_function_fn, fn, types, args, kwargs)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if tx.output.torch_function_enabled:
            import torch
            from .builder import SourcelessBuilder, VariableBuilder
            if _is_attr_overidden(tx, self, name):
                unimplemented(f'Calling overridden method {name} on a tensor subclass with a __torch_function__ override is not supported')
            if self.source:
                func_var = VariableBuilder(tx, AttrSource(AttrSource(self.source, '__class__'), name))(inspect.getattr_static(self.python_type(), name))
            else:
                func_var = SourcelessBuilder()(tx, getattr(torch.Tensor, name))
            return dispatch_torch_function(tx, func_var, [self] + args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)