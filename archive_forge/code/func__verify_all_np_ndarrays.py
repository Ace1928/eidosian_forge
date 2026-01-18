import os as _os
import ctypes
import numpy as _np  # pylint: disable=unused-import
from ._internal import NDArrayBase, _imperative_invoke # pylint: disable=unused-import
from ..ndarray_doc import _build_doc
from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null, _is_np_op, _output_is_list  # pylint: disable=unused-import
from ..util import use_np_shape  # pylint: disable=unused-import
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
def _verify_all_np_ndarrays(op_name, func_name, args, out):
    """Verify if all the arrays are numpy ndarrays.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    args : list of arrays
        Input ndarray arguments to be checked.
    out : ndarray or None or list of ndarrays
        User-provided output ndarrays.
    """
    from ..numpy import ndarray as np_ndarray
    for arr in args:
        if arr is not None and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. This is a numpy operator which can only accept MXNet numpy ndarrays, while received a legacy ndarray. Please ensure that you have activated numpy semantics by calling `npx.set_np()` in your code. If you still see this error with numpy semantics activated, please call `as_np_ndarray()` upon the legacy ndarray to convert it to an MXNet numpy ndarray, and then feed the converted array to this operator.'.format(op_name, func_name))
    if out is None:
        return
    if not isinstance(out, (list, tuple)):
        out = [out]
    for arr in out:
        if arr is not None and (not isinstance(arr, np_ndarray)):
            raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. This is a numpy operator which can only accept MXNet numpy ndarrays, while received a legacy ndarray. Please ensure that you have activated numpy semantics by calling `npx.set_np()` in your code. If you still see this error with numpy semantics activated, please call `as_np_ndarray()` upon the legacy ndarray to convert it to an MXNet numpy ndarray, and then feed the converted array to this operator.'.format(op_name, func_name))