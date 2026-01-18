from __future__ import (absolute_import, division, print_function)
import copy
from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs
class NativeOdeintSys(_NativeSysBase):
    _NativeCode = NativeOdeintCode
    _native_name = 'odeint'