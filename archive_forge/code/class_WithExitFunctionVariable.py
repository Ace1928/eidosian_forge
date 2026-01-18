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
class WithExitFunctionVariable(VariableTracker):

    def __init__(self, ctx: ContextWrappingVariable, target, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(ctx, ContextWrappingVariable)
        self.ctx = ctx
        self.target = target

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen):
        output = AttrSource(codegen.tx.import_source(self.ctx.module_name()), self.ctx.fn_name()).reconstruct(codegen)
        if codegen.tx.output.partial_convert:
            loads = [codegen.create_load_const(val) for val in self.ctx.target_values]
            output.extend(loads)
            output.extend([*create_call_function(len(loads), True), create_instruction('SETUP_WITH', target=self.target), create_instruction('POP_TOP')])
        return output