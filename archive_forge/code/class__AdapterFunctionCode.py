import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class _AdapterFunctionCode(CallableObjectProxy):

    def __init__(self, wrapped_code, adapter_code):
        super(_AdapterFunctionCode, self).__init__(wrapped_code)
        self._self_adapter_code = adapter_code

    @property
    def co_argcount(self):
        return self._self_adapter_code.co_argcount

    @property
    def co_code(self):
        return self._self_adapter_code.co_code

    @property
    def co_flags(self):
        return self._self_adapter_code.co_flags

    @property
    def co_kwonlyargcount(self):
        return self._self_adapter_code.co_kwonlyargcount

    @property
    def co_varnames(self):
        return self._self_adapter_code.co_varnames