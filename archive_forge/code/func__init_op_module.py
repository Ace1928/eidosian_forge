import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _init_op_module(root_namespace, module_name, make_op_func):
    """
    Registers op functions created by `make_op_func` under
    `root_namespace.module_name.[submodule_name]`,
    where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.

    Parameters
    ----------
    root_namespace : str
        Top level module name, `mxnet` in the current cases.
    module_name : str
        Second level module name, `ndarray` and `symbol` in the current cases.
    make_op_func : function
        Function for creating op functions for `ndarray` and `symbol` modules.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_name = py_str(plist[i])
        if not _is_np_op(op_name):
            op_names.append(op_name)
    module_op = sys.modules['%s.%s.op' % (root_namespace, module_name)]
    module_internal = sys.modules['%s.%s._internal' % (root_namespace, module_name)]
    contrib_module_name_old = '%s.contrib.%s' % (root_namespace, module_name)
    contrib_module_old = sys.modules[contrib_module_name_old]
    submodule_dict = {}
    for op_name_prefix in _OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] = sys.modules['%s.%s.%s' % (root_namespace, module_name, op_name_prefix[1:-1])]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        op_name_prefix = _get_op_name_prefix(name)
        module_name_local = module_name
        if len(op_name_prefix) > 0:
            if op_name_prefix != '_random_' or name.endswith('_like'):
                func_name = name[len(op_name_prefix):]
                cur_module = submodule_dict[op_name_prefix]
                module_name_local = '%s.%s.%s' % (root_namespace, module_name, op_name_prefix[1:-1])
            else:
                func_name = name
                cur_module = module_internal
        elif name.startswith('_'):
            func_name = name
            cur_module = module_internal
        else:
            func_name = name
            cur_module = module_op
        function = make_op_func(hdl, name, func_name)
        function.__module__ = module_name_local
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)
        if op_name_prefix == '_contrib_':
            hdl = OpHandle()
            check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
            func_name = name[len(op_name_prefix):]
            function = make_op_func(hdl, name, func_name)
            function.__module__ = contrib_module_name_old
            setattr(contrib_module_old, function.__name__, function)
            contrib_module_old.__all__.append(function.__name__)