from array import array
import ctypes
import warnings
from ..ndarray import NDArray
from ..base import _LIB, c_str_array, c_handle_array, c_array, c_array_buf, c_str
from ..base import check_call, string_types
from ..base import KVStoreHandle
from ..profiler import set_kvstore_handle
def _ctype_key_value(keys, vals):
    """Returns ctype arrays for the key-value args, and the whether string keys are used.
    For internal use only.
    """
    if isinstance(keys, (tuple, list)):
        assert len(keys) == len(vals)
        c_keys = []
        c_vals = []
        use_str_keys = None
        for key, val in zip(keys, vals):
            c_key_i, c_val_i, str_keys_i = _ctype_key_value(key, val)
            c_keys += c_key_i
            c_vals += c_val_i
            use_str_keys = str_keys_i if use_str_keys is None else use_str_keys
            assert use_str_keys == str_keys_i, 'inconsistent types of keys detected.'
        c_keys_arr = c_array(ctypes.c_char_p, c_keys) if use_str_keys else c_array(ctypes.c_int, c_keys)
        c_vals_arr = c_array(ctypes.c_void_p, c_vals)
        return (c_keys_arr, c_vals_arr, use_str_keys)
    assert isinstance(keys, (int,) + string_types), 'unexpected type for keys: ' + str(type(keys))
    use_str_keys = isinstance(keys, string_types)
    if isinstance(vals, NDArray):
        c_keys = c_str_array([keys]) if use_str_keys else c_array_buf(ctypes.c_int, array('i', [keys]))
        return (c_keys, c_handle_array([vals]), use_str_keys)
    else:
        for value in vals:
            assert isinstance(value, NDArray)
        c_keys = c_str_array([keys] * len(vals)) if use_str_keys else c_array_buf(ctypes.c_int, array('i', [keys] * len(vals)))
        return (c_keys, c_handle_array(vals), use_str_keys)