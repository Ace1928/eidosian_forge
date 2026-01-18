import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
def _method_reply_error(connection, message, exception):
    name = getattr(exception, '_dbus_error_name', None)
    if name is not None:
        pass
    elif getattr(exception, '__module__', '') in ('', '__main__'):
        name = 'org.freedesktop.DBus.Python.%s' % exception.__class__.__name__
    else:
        name = 'org.freedesktop.DBus.Python.%s.%s' % (exception.__module__, exception.__class__.__name__)
    et, ev, etb = sys.exc_info()
    if isinstance(exception, DBusException) and (not exception.include_traceback):
        contents = exception.get_dbus_message()
    elif ev is exception:
        contents = ''.join(traceback.format_exception(et, ev, etb))
    else:
        contents = ''.join(traceback.format_exception_only(exception.__class__, exception))
    reply = ErrorMessage(message, name, contents)
    if not message.get_no_reply():
        connection.send_message(reply)