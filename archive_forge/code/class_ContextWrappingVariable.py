import dataclasses
import inspect
from typing import Callable, Dict, List, Optional
import torch._C
from torch._guards import Guard
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..device_interface import get_interface_for_device
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from .base import VariableTracker
from .functions import (
class ContextWrappingVariable(VariableTracker):
    _nonvar_fields = {'cm_obj', 'target_values', 'initial_values', 'state', *VariableTracker._nonvar_fields}

    def __init__(self, target_values, initial_values=None, *, state=None, **kwargs):
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values
        self.state = ContextMangerState() if state is None else state

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        self.set_cleanup_hook(tx)
        return variables.ConstantVariable.create(None)

    def set_cleanup_hook(self, tx, fn=None):
        if fn is None:

            def fn():
                self._call_func(tx, self.initial_values)
        self.state.cleanup_fn = fn
        tx.output.add_cleanup_hook(self.state.cleanup)

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        return variables.ConstantVariable.create(None)

    def reconstruct(self, codegen):
        attr_source = AttrSource(codegen.tx.import_source(self.module_name()), self.fn_name())
        return attr_source.reconstruct(codegen)

    def module_name(self):
        raise NotImplementedError('module_name called on base')

    def fn_name(self):
        raise NotImplementedError('fn_name called on base')

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        assert len(args) == 1
        if isinstance(args[0], NestedUserFunctionVariable):
            args[0] = UserFunctionVariable(args[0].get_function())
        assert isinstance(args[0], (UserMethodVariable, UserFunctionVariable))
        if isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)
        if isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)