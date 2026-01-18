import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
def __async_result_handler(self, obj, result, user_data):
    result_callback, error_callback, real_user_data = user_data
    try:
        ret = obj.call_finish(result)
    except Exception:
        etype, e = sys.exc_info()[:2]
        if error_callback:
            error_callback(obj, e, real_user_data)
        else:
            result_callback(obj, e, real_user_data)
        return
    result_callback(obj, self._unpack_result(ret), real_user_data)