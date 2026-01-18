import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def _Introspect(self):
    kwargs = {}
    return self._bus.call_async(self._named_service, self.__dbus_object_path__, INTROSPECTABLE_IFACE, 'Introspect', '', (), self._introspect_reply_handler, self._introspect_error_handler, require_main_loop=False, **kwargs)