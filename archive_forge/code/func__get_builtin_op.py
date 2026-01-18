import inspect
from . import _numpy_op_doc
from . import numpy as mx_np
from . import numpy_extension as mx_npx
from .base import _NP_OP_SUBMODULE_LIST, _NP_EXT_OP_SUBMODULE_LIST, _get_op_submodule_name
def _get_builtin_op(op_name):
    if op_name.startswith('_np_'):
        root_module = mx_np
        op_name_prefix = '_np_'
        submodule_name_list = _NP_OP_SUBMODULE_LIST
    elif op_name.startswith('_npx_'):
        root_module = mx_npx
        op_name_prefix = '_npx_'
        submodule_name_list = _NP_EXT_OP_SUBMODULE_LIST
    else:
        return None
    submodule_name = _get_op_submodule_name(op_name, op_name_prefix, submodule_name_list)
    op_module = root_module
    if len(submodule_name) > 0:
        op_module = getattr(root_module, submodule_name[1:-1], None)
        if op_module is None:
            raise ValueError('Cannot find submodule {} in module {}'.format(submodule_name[1:-1], root_module.__name__))
    op = getattr(op_module, op_name[len(op_name_prefix) + len(submodule_name):], None)
    if op is None:
        raise ValueError('Cannot find operator {} in module {}'.format(op_name[len(op_name_prefix):], root_module.__name__))
    return op