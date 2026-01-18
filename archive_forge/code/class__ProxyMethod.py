import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
class _ProxyMethod:
    """A proxy method.

    Typically a member of a ProxyObject. Calls to the
    method produce messages that travel over the Bus and are routed
    to a specific named Service.
    """

    def __init__(self, proxy, connection, bus_name, object_path, method_name, iface):
        if object_path == LOCAL_PATH:
            raise DBusException('Methods may not be called on the reserved path %s' % LOCAL_PATH)
        self._proxy = proxy
        self._connection = connection
        self._named_service = bus_name
        self._object_path = object_path
        _dbus_bindings.validate_member_name(method_name)
        self._method_name = method_name
        if iface is not None:
            _dbus_bindings.validate_interface_name(iface)
        self._dbus_interface = iface

    def __call__(self, *args, **keywords):
        reply_handler = keywords.pop('reply_handler', None)
        error_handler = keywords.pop('error_handler', None)
        ignore_reply = keywords.pop('ignore_reply', False)
        signature = keywords.pop('signature', None)
        if reply_handler is not None or error_handler is not None:
            if reply_handler is None:
                raise MissingReplyHandlerException()
            elif error_handler is None:
                raise MissingErrorHandlerException()
            elif ignore_reply:
                raise TypeError('ignore_reply and reply_handler cannot be used together')
        dbus_interface = keywords.pop('dbus_interface', self._dbus_interface)
        if signature is None:
            if dbus_interface is None:
                key = self._method_name
            else:
                key = dbus_interface + '.' + self._method_name
            signature = self._proxy._introspect_method_map.get(key, None)
        if ignore_reply or reply_handler is not None:
            self._connection.call_async(self._named_service, self._object_path, dbus_interface, self._method_name, signature, args, reply_handler, error_handler, **keywords)
        else:
            return self._connection.call_blocking(self._named_service, self._object_path, dbus_interface, self._method_name, signature, args, **keywords)

    def call_async(self, *args, **keywords):
        reply_handler = keywords.pop('reply_handler', None)
        error_handler = keywords.pop('error_handler', None)
        signature = keywords.pop('signature', None)
        dbus_interface = keywords.pop('dbus_interface', self._dbus_interface)
        if signature is None:
            if dbus_interface:
                key = dbus_interface + '.' + self._method_name
            else:
                key = self._method_name
            signature = self._proxy._introspect_method_map.get(key, None)
        self._connection.call_async(self._named_service, self._object_path, dbus_interface, self._method_name, signature, args, reply_handler, error_handler, **keywords)