import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _init_np_op_module(root_module_name, np_module_name, mx_module_name, make_op_func):
    """
    Register numpy operators in namespaces `mxnet.numpy`, `mxnet.ndarray.numpy`
    and `mxnet.symbol.numpy`. They are used in imperative mode, Gluon APIs w/o hybridization,
    and Gluon APIs w/ hybridization, respectively. Essentially, operators with the same name
    registered in three namespaces, respectively share the same functionality in C++ backend.
    Different namespaces are needed for dispatching operator calls in Gluon's `HybridBlock` by `F`.

    Parameters
    ----------
    root_module_name : str
        Top level module name, `mxnet` in the current cases.
    np_module_name : str
        Second level module name, `numpy` or `numpy_extension` in the current case.
    make_op_func : function
        Function for creating op functions.
    """
    from . import _numpy_op_doc as _np_op_doc
    if np_module_name == 'numpy':
        op_name_prefix = _NP_OP_PREFIX
        submodule_name_list = _NP_OP_SUBMODULE_LIST
    elif np_module_name == 'numpy_extension':
        op_name_prefix = _NP_EXT_OP_PREFIX
        submodule_name_list = _NP_EXT_OP_SUBMODULE_LIST
    elif np_module_name == 'numpy._internal':
        op_name_prefix = _NP_INTERNAL_OP_PREFIX
        submodule_name_list = []
    else:
        raise ValueError('unsupported np module name {}'.format(np_module_name))
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        name = py_str(plist[i])
        if name.startswith(op_name_prefix):
            op_names.append(name)
    if mx_module_name is None:
        op_module_name = '%s.%s._op' % (root_module_name, np_module_name)
        op_submodule_name = '%s.%s' % (root_module_name, np_module_name)
    elif mx_module_name in ('ndarray', 'symbol'):
        op_module_name = '%s.%s.%s' % (root_module_name, mx_module_name, np_module_name)
        if op_name_prefix != _NP_INTERNAL_OP_PREFIX:
            op_module_name += '._op'
        op_submodule_name = '%s.%s.%s' % (root_module_name, mx_module_name, np_module_name)
    else:
        raise ValueError('unsupported mxnet module {}'.format(mx_module_name))
    op_submodule_name += '.%s'
    op_module = sys.modules[op_module_name]
    submodule_dict = {}
    for submodule_name in submodule_name_list:
        submodule_dict[submodule_name] = sys.modules[op_submodule_name % submodule_name[1:-1]]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        submodule_name = _get_op_submodule_name(name, op_name_prefix, submodule_name_list)
        if len(submodule_name) > 0:
            func_name = name[len(op_name_prefix) + len(submodule_name):]
            cur_module = submodule_dict[submodule_name]
            module_name_local = op_submodule_name % submodule_name[1:-1]
        else:
            func_name = name[len(op_name_prefix):]
            cur_module = op_module
            module_name_local = op_module_name[:-len('._op')] if op_module_name.endswith('._op') else op_module_name
        function = make_op_func(hdl, name, func_name)
        function.__module__ = module_name_local
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)
        if hasattr(_np_op_doc, name):
            function.__doc__ = getattr(_np_op_doc, name).__doc__
        else:
            function.__doc__ = re.sub('NDArray', 'ndarray', function.__doc__)