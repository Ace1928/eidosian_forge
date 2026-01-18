import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def _set_updater(self, updater):
    """Sets a push updater into the store.

        This function only changes the local store. When running on multiple machines one must
        use `set_optimizer`.

        Parameters
        ----------
        updater : function
            The updater function.

        Examples
        --------
        >>> def update(key, input, stored):
        ...     print "update on key: %d" % key
        ...     stored += input * 2
        >>> kv._set_updater(update)
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> kv.push('3', mx.nd.ones(shape))
        update on key: 3
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
        """
    self._updater = updater
    _updater_proto = ctypes.CFUNCTYPE(None, ctypes.c_int, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
    self._updater_func = _updater_proto(_updater_wrapper(updater))
    _str_updater_proto = ctypes.CFUNCTYPE(None, ctypes.c_char_p, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
    self._str_updater_func = _str_updater_proto(_updater_wrapper(updater))
    check_call(_LIB.MXKVStoreSetUpdaterEx(self.handle, self._updater_func, self._str_updater_func, None))