import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def _introspect_reply_handler(self, data):
    self._introspect_lock.acquire()
    try:
        try:
            self._introspect_method_map = process_introspection_data(data)
        except IntrospectionParserException as e:
            self._introspect_error_handler(e)
            return
        self._introspect_state = self.INTROSPECT_STATE_INTROSPECT_DONE
        self._pending_introspect = None
        self._introspect_execute_queue()
    finally:
        self._introspect_lock.release()