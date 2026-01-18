import os
import sys
import ctypes
from ..base import _LIB, check_call
from .base import py_str, c_str
def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]
    for name in list_global_func_names():
        if prefix == 'api':
            fname = name
            if name.startswith('_'):
                target_module = sys.modules['mxnet._api_internal']
            else:
                target_module = module
        else:
            if not name.startswith(prefix):
                continue
            fname = name[len(prefix) + 1:]
            target_module = module
        if fname.find('.') != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = 'MXNet PackedFunc %s. ' % fname
        setattr(target_module, ff.__name__, ff)