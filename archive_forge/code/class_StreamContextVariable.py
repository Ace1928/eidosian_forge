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
class StreamContextVariable(ContextWrappingVariable):

    @staticmethod
    def create(tx, target_value, **kwargs):
        from .builder import wrap_fx_proxy_cls
        current_stream_method = get_interface_for_device(target_value.device).current_stream
        current_stream = wrap_fx_proxy_cls(StreamVariable, tx, tx.output.create_proxy('call_function', current_stream_method, (None,), {}))
        return StreamContextVariable(target_values=[target_value], initial_values=[current_stream], device=target_value.device, **kwargs)

    def __init__(self, target_values, device, initial_values=None, **kwargs):
        super().__init__(target_values=target_values, initial_values=initial_values, **kwargs)
        self.device = device
        self.set_stream = get_interface_for_device(self.device).set_stream
        self.set_stream_id = get_interface_for_device(self.device)._set_stream_by_id

    def enter(self, tx):
        if self.target_values[0].as_proxy() is not None:
            tx.output.create_proxy('call_function', self.set_stream, (self.target_values[0].as_proxy(),), {})
        else:
            stream = self.target_values[0].value
            tx.output.create_proxy('call_function', self.set_stream_id, (stream.stream_id, stream.device_index, stream.device_type), {})
        self.set_stream(self.target_values[0].value)
        self.set_cleanup_hook(tx, lambda: self.set_stream(self.initial_values[0].value))

    def exit(self, tx, *args):
        tx.output.create_proxy('call_function', self.set_stream, (self.initial_values[0].as_proxy(),), {})
        self.state.cleanup_assert()

    def module_name(self):
        return 'torch.' + str(self.device)

    def fn_name(self):
        return 'stream'