import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
class NameOwnerWatch(object):
    __slots__ = ('_match', '_pending_call')

    def __init__(self, bus_conn, bus_name, callback):
        validate_bus_name(bus_name)

        def signal_cb(owned, old_owner, new_owner):
            callback(new_owner)

        def error_cb(e):
            if e.get_dbus_name() == _NAME_HAS_NO_OWNER:
                callback('')
            else:
                logging.basicConfig()
                _logger.debug('GetNameOwner(%s) failed:', bus_name, exc_info=(e.__class__, e, None))
        self._match = bus_conn.add_signal_receiver(signal_cb, 'NameOwnerChanged', BUS_DAEMON_IFACE, BUS_DAEMON_NAME, BUS_DAEMON_PATH, arg0=bus_name)
        self._pending_call = bus_conn.call_async(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'GetNameOwner', 's', (bus_name,), callback, error_cb)

    def cancel(self):
        if self._match is not None:
            self._match.remove()
        if self._pending_call is not None:
            self._pending_call.cancel()
        self._match = None
        self._pending_call = None