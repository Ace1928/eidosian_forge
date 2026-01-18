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
class StreamVariable(VariableTracker):

    def __init__(self, proxy, value, device, **kwargs):
        if proxy is not None and 'example_value' in proxy.node.meta:
            assert proxy.node.meta['example_value'] == value
        assert value.device.type == device, 'stream value is not equal to the passed device'
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.device = device

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        assert hasattr(self.value, name), f'no stream method found named {name}'
        assert name in ['wait_stream', 'synchronize', 'query', 'record_event', 'wait_event'], f' unsupported stream method {name}'
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls
        if name in ('wait_stream', 'synchronize', 'wait_event'):
            tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + args, kwargs))
            return variables.ConstantVariable(None)
        elif name == 'query':
            return wrap_fx_proxy_cls(target_cls=variables.ConstantVariable, tx=tx, proxy=tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + args, kwargs)))
        elif name == 'record_event':
            return wrap_fx_proxy_cls(target_cls=EventVariable, tx=tx, proxy=tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + args, kwargs)))
        else:
            unimplemented(self.device + ' stream method ' + name + ' unsupported')

    def as_proxy(self):
        return self.proxy