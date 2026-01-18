import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
class DBusProxy(Gio.DBusProxy):
    """Provide comfortable and pythonic method calls.

    This marshalls the method arguments into a GVariant, invokes the
    call_sync() method on the DBusProxy object, and unmarshalls the result
    GVariant back into a Python tuple.

    The first argument always needs to be the D-Bus signature tuple of the
    method call. Example:

      proxy = Gio.DBusProxy.new_sync(...)
      result = proxy.MyMethod('(is)', 42, 'hello')

    The exception are methods which take no arguments, like
    proxy.MyMethod('()'). For these you can omit the signature and just write
    proxy.MyMethod().

    Optional keyword arguments:

    - timeout: timeout for the call in milliseconds (default to D-Bus timeout)

    - flags: Combination of Gio.DBusCallFlags.*

    - result_handler: Do an asynchronous method call and invoke
         result_handler(proxy_object, result, user_data) when it finishes.

    - error_handler: If the asynchronous call raises an exception,
      error_handler(proxy_object, exception, user_data) is called when it
      finishes. If error_handler is not given, result_handler is called with
      the exception object as result instead.

    - user_data: Optional user data to pass to result_handler for
      asynchronous calls.

    Example for asynchronous calls:

      def mymethod_done(proxy, result, user_data):
          if isinstance(result, Exception):
              # handle error
          else:
              # do something with result

      proxy.MyMethod('(is)', 42, 'hello',
          result_handler=mymethod_done, user_data='data')
    """

    def __getattr__(self, name):
        return _DBusProxyMethodCall(self, name)