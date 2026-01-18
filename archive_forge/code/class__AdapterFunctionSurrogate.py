import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class _AdapterFunctionSurrogate(CallableObjectProxy):

    def __init__(self, wrapped, adapter):
        super(_AdapterFunctionSurrogate, self).__init__(wrapped)
        self._self_adapter = adapter

    @property
    def __code__(self):
        return _AdapterFunctionCode(self.__wrapped__.__code__, self._self_adapter.__code__)

    @property
    def __defaults__(self):
        return self._self_adapter.__defaults__

    @property
    def __kwdefaults__(self):
        return self._self_adapter.__kwdefaults__

    @property
    def __signature__(self):
        if 'signature' not in globals():
            return self._self_adapter.__signature__
        else:
            return signature(self._self_adapter)
    if PY2:
        func_code = __code__
        func_defaults = __defaults__