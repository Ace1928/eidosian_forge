import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class AdapterWrapper(FunctionWrapper):
    __bound_function_wrapper__ = _BoundAdapterWrapper

    def __init__(self, *args, **kwargs):
        adapter = kwargs.pop('adapter')
        super(AdapterWrapper, self).__init__(*args, **kwargs)
        self._self_surrogate = _AdapterFunctionSurrogate(self.__wrapped__, adapter)
        self._self_adapter = adapter

    @property
    def __code__(self):
        return self._self_surrogate.__code__

    @property
    def __defaults__(self):
        return self._self_surrogate.__defaults__

    @property
    def __kwdefaults__(self):
        return self._self_surrogate.__kwdefaults__
    if PY2:
        func_code = __code__
        func_defaults = __defaults__

    @property
    def __signature__(self):
        return self._self_surrogate.__signature__