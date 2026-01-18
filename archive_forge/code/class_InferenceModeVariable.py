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
class InferenceModeVariable(ContextWrappingVariable):

    @staticmethod
    def create(tx, target_values, **kwargs):
        var = InferenceModeVariable(target_values, initial_values=torch.is_inference_mode_enabled(), **kwargs)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        if initial_values is None:
            initial_values = torch.is_inference_mode_enabled()
        super().__init__(target_values=target_values, initial_values=initial_values, **kwargs)
        self.target_values = target_values

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        tx.output.create_node('call_function', torch.autograd.grad_mode._exit_inference_mode, (self.state.proxy,), {})

    def enter(self, tx):
        ctx = torch.autograd.grad_mode._enter_inference_mode(self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.autograd.grad_mode._exit_inference_mode(ctx))
        self.state.proxy = tx.output.create_node('call_function', torch.autograd.grad_mode._enter_inference_mode, (self.target_values,), {})

    def module_name(self):
        return 'torch.inference_mode'

    def fn_name(self):
        return 'inference_mode'