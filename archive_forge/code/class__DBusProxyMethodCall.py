import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
class _DBusProxyMethodCall:
    """Helper class to implement DBusProxy method calls."""

    def __init__(self, dbus_proxy, method_name):
        self.dbus_proxy = dbus_proxy
        self.method_name = method_name

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

    def __call__(self, *args, **kwargs):
        if args:
            signature = args[0]
            args = args[1:]
            if not isinstance(signature, str):
                raise TypeError('first argument must be the method signature string: %r' % signature)
        else:
            signature = '()'
        arg_variant = GLib.Variant(signature, tuple(args))
        if 'result_handler' in kwargs:
            user_data = (kwargs['result_handler'], kwargs.get('error_handler'), kwargs.get('user_data'))
            self.dbus_proxy.call(self.method_name, arg_variant, kwargs.get('flags', 0), kwargs.get('timeout', -1), None, self.__async_result_handler, user_data)
        else:
            result = self.dbus_proxy.call_sync(self.method_name, arg_variant, kwargs.get('flags', 0), kwargs.get('timeout', -1), None)
            return self._unpack_result(result)

    @classmethod
    def _unpack_result(klass, result):
        """Convert a D-BUS return variant into an appropriate return value"""
        result = result.unpack()
        if len(result) == 1:
            result = result[0]
        elif len(result) == 0:
            result = None
        return result