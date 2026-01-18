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
class EventVariable(VariableTracker):

    def __init__(self, proxy, value, **kwargs):
        if proxy is not None and 'example_value' in proxy.node.meta:
            assert proxy.node.meta['example_value'] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls
        if name in ('wait', 'record', 'synchronize'):
            tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + args, kwargs))
            return variables.ConstantVariable(None)
        elif name == 'query':
            return wrap_fx_proxy_cls(target_cls=variables.ConstantVariable, tx=tx, proxy=tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + args, kwargs)))
        else:
            unimplemented(f'event method {name} unsupported')

    def as_proxy(self):
        return self.proxy