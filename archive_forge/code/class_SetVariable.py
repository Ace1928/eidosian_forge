import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
class SetVariable(VariableTracker):

    @dataclasses.dataclass
    class SetElement:
        vt: VariableTracker
        underlying_value: Any

        def __hash__(self) -> int:
            return hash(self.underlying_value)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, SetVariable.SetElement):
                return False
            if isinstance(self.vt, variables.TensorVariable):
                return self.underlying_value is other.underlying_value
            else:
                return self.underlying_value == other.underlying_value

    def __init__(self, items: List[VariableTracker], **kwargs):
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all((isinstance(x, VariableTracker) for x in items))
        self.items = []
        self._add(items)

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def python_type(self):
        return set

    def reconstruct(self, codegen):
        codegen.load_import_from('builtins', 'set')
        codegen.foreach(self.items)
        return [create_instruction('BUILD_SET', arg=len(self.items))] + create_call_function(1, True)

    def _as_set_element(self, vt):
        from .base import VariableTracker
        from .misc import MethodWrapperVariable
        from .tensor import TensorVariable
        assert isinstance(vt, VariableTracker)
        if isinstance(vt, TensorVariable):
            fake_tensor = vt.as_proxy().node.meta.get('example_value')
            if fake_tensor is None:
                unimplemented('Cannot check Tensor object identity without its fake value')
            return SetVariable.SetElement(vt, fake_tensor)
        if isinstance(vt, ConstantVariable):
            return SetVariable.SetElement(vt, vt.value)
        if isinstance(vt, MethodWrapperVariable):
            return SetVariable.SetElement(vt, vt.as_python_constant())
        unimplemented(f'Sets with {type(vt)} NYI')

    @property
    def _underlying_items(self):
        underlying_items = set()
        for current_item in self.items:
            assert current_item not in underlying_items, 'Items modeling set invariant violated'
            underlying_items.add(self._as_set_element(current_item))
        return underlying_items

    def _add(self, item):
        underlying_items = self._underlying_items
        if isinstance(item, (list, set)):
            items_to_add = item
        else:
            items_to_add = [item]
        for item_to_add in items_to_add:
            set_element = self._as_set_element(item_to_add)
            if set_element not in underlying_items:
                underlying_items.add(set_element)
                self.items.append(set_element.vt)
            else:
                for e in underlying_items:
                    if hash(set_element) == hash(e):
                        alias_guard = make_dupe_guard(e.vt.source, set_element.vt.source)
                        if alias_guard:
                            install_guard(e.vt.source.make_guard(alias_guard))
        return self.items

    def call_method(self, tx, name, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> 'VariableTracker':
        if name == 'add' and args and self.mutable_local:
            assert not kwargs
            item = args[0]
            result = SetVariable(self._add(item), mutable_local=self.mutable_local)
            tx.replace_all(self, result)
            return ConstantVariable.create(None)
        elif name == 'pop' and self.mutable_local:
            assert not kwargs
            assert not args
            items = list(self.items)
            result = items.pop()
            tx.replace_all(self, SetVariable(items))
            return result
        elif name == '__len__':
            return ConstantVariable.create(len(self.items))
        elif name == '__contains__':
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items, args[0], tx, check_tensor_identity=True)
        else:
            return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, arg: VariableTracker):
        raise RuntimeError('Illegal to getitem on a set')

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return list(self.items)