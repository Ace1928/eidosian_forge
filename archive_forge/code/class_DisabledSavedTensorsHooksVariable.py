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
class DisabledSavedTensorsHooksVariable(ContextWrappingVariable):
    """represents torch.autograd.graph.disable_saved_tensors_hook."""

    @staticmethod
    def create(tx, target_value, **kwargs):
        var = DisabledSavedTensorsHooksVariable(target_values=[target_value], initial_values=[torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()], **kwargs)
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(target_values=target_values, initial_values=initial_values, **kwargs)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        if value is not None:
            tx.output.create_node('call_function', torch._C._autograd._saved_tensors_hooks_disable, (value,), {})
            torch._C._autograd._saved_tensors_hooks_disable(value)
        else:
            tx.output.create_node('call_function', torch._C._autograd._saved_tensors_hooks_enable, (), {})
            torch._C._autograd._saved_tensors_hooks_enable()

    def module_name(self):
        return 'torch.autograd.graph'

    def fn_name(self):
        return 'disable_saved_tensors_hooks'