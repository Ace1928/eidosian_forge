import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
class BaseListVariable(VariableTracker):

    @staticmethod
    def cls_for_instance(obj):
        if is_namedtuple(obj):
            return functools.partial(NamedTupleVariable, tuple_cls=type(obj))
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj):
        return {iter: ListIteratorVariable, list: ListVariable, slice: SliceVariable, torch.Size: SizeVariable, tuple: TupleVariable, odict_values: ListVariable, torch.nn.ParameterList: ListVariable, torch.nn.ModuleList: ListVariable, collections.deque: DequeVariable}[obj]

    def __init__(self, items: List[VariableTracker], **kwargs):
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all((isinstance(x, VariableTracker) for x in items))
        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def modified(self, items, **kwargs):
        return type(self)(items, **kwargs)

    @property
    def value(self):
        return self.as_python_constant()

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
        from .tensor import SymNodeVariable
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()
        if isinstance(index, slice):
            if self.source is not None:
                return self.clone(items=self.items[index], source=GetItemSource(self.source, index), mutable_local=MutableLocal() if self.mutable_local else None)
            else:
                return self.clone(items=self.items[index], mutable_local=MutableLocal() if self.mutable_local else None)
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name == '__getitem__':
            from .tensor import TensorVariable
            assert not kwargs and len(args) == 1
            if isinstance(args[0], TensorVariable):
                value = get_fake_value(args[0].as_proxy().node, tx)
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    unimplemented('__getitem__ with non-constant tensor')
            else:
                value = args[0]
            return self.getitem_const(value)
        elif name == '__contains__':
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items, args[0], tx)
        elif name == 'index':
            from .builder import SourcelessBuilder
            return tx.inline_user_function_return(SourcelessBuilder()(tx, polyfill.index), [self] + list(args), kwargs)
        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def list_compare(tx, op, left, right):
        from .builtin import BuiltinVariable
        eq_result = BaseListVariable.list_eq(tx, left, right)
        if op is operator.eq:
            return eq_result
        elif op is operator.ne:
            return BuiltinVariable(operator.not_).call_function(tx, [eq_result], {})
        else:
            unimplemented(f'list_compare {left} {op} {right}')

    @staticmethod
    def list_eq(tx, left, right):
        from .builtin import BuiltinVariable
        if len(left.items) != len(right.items):
            return ConstantVariable.create(False)
        if len(left.items) == 0:
            return ConstantVariable.create(True)
        comps = []
        for l, r in zip(left.items, right.items):
            comp = BuiltinVariable(operator.eq).call_function(tx, [l, r], {})
            if comp.is_python_constant() and (not comp.as_python_constant()):
                return comp
            comps.append(comp)
        return functools.reduce(lambda a, b: BuiltinVariable(operator.and_).call_function(tx, [a, b], {}), comps)