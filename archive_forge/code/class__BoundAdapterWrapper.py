import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class _BoundAdapterWrapper(BoundFunctionWrapper):

    @property
    def __func__(self):
        return _AdapterFunctionSurrogate(self.__wrapped__.__func__, self._self_parent._self_adapter)

    @property
    def __signature__(self):
        if 'signature' not in globals():
            return self.__wrapped__.__signature__
        else:
            return signature(self._self_parent._self_adapter)
    if PY2:
        im_func = __func__