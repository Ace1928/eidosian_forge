import os as _os
import ctypes
import numpy as _np  # pylint: disable=unused-import
from ._internal import NDArrayBase, _imperative_invoke # pylint: disable=unused-import
from ..ndarray_doc import _build_doc
from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null, _is_np_op, _output_is_list  # pylint: disable=unused-import
from ..util import use_np_shape  # pylint: disable=unused-import
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
def _make_ndarray_function(handle, name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code, doc_str = _generate_ndarray_function_code(handle, name, func_name)
    local = {}
    exec(code, None, local)
    ndarray_function = local[func_name]
    ndarray_function.__name__ = func_name
    ndarray_function.__doc__ = doc_str
    ndarray_function.__module__ = 'mxnet.ndarray'
    return ndarray_function