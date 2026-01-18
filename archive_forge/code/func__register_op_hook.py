import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle, CachedOpHandle
from ..base import check_call
from .. import _global_var
def _register_op_hook(self, callback, monitor_all=False):
    """Install callback for monitor.

        Parameters
        ----------
        callback : function
            Takes a string for node_name, string for op_name and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input _imperative_invoked output, otherwise monitor output only.
        """
    cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
    if callback:
        self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
    check_call(_LIB.MXCachedOpRegisterOpHook(self.handle, self._monitor_callback, ctypes.c_int(monitor_all)))