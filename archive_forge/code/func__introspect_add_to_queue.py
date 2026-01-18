import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def _introspect_add_to_queue(self, callback, args, kwargs):
    self._introspect_lock.acquire()
    try:
        if self._introspect_state == self.INTROSPECT_STATE_INTROSPECT_IN_PROGRESS:
            self._pending_introspect_queue.append((callback, args, kwargs))
        else:
            callback(*args, **kwargs)
    finally:
        self._introspect_lock.release()