from __future__ import (absolute_import, division, print_function)
import copy
import os
from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs
class NativeGSLSys(_NativeSysBase):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'