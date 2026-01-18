import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle, CachedOpHandle
from ..base import check_call
from .. import _global_var
def _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op, output_is_list):
    """ctypes implementation of imperative invoke wrapper"""
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            out = (out,)
        num_output = ctypes.c_int(len(out))
        output_vars = c_handle_array(out)
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else:
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)
    out_stypes = ctypes.POINTER(ctypes.c_int)()
    check_call(_LIB.MXImperativeInvokeEx(ctypes.c_void_p(handle), ctypes.c_int(len(ndargs)), c_handle_array(ndargs), ctypes.byref(num_output), ctypes.byref(output_vars), ctypes.c_int(len(keys)), c_str_array(keys), c_str_array([str(s) for s in vals]), ctypes.byref(out_stypes)))
    create_ndarray_fn = _global_var._np_ndarray_cls if is_np_op else _global_var._ndarray_cls
    if original_output is not None:
        return original_output
    if num_output.value == 1 and (not output_is_list):
        return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle), stype=out_stypes[0])
    else:
        return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle), stype=out_stypes[i]) for i in range(num_output.value)]