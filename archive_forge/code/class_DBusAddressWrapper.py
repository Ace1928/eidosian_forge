import os
from typing import Any, List, Tuple
from jeepney import (
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT, \
from secretstorage.dhcrypto import Session, int_to_bytes
from secretstorage.exceptions import ItemNotFoundException, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
class DBusAddressWrapper(DBusAddress):
    """A wrapper class around :class:`jeepney.wrappers.DBusAddress`
    that adds some additional methods for calling and working with
    properties, and converts error responses to SecretStorage
    exceptions.

    .. versionadded:: 3.0
    """

    def __init__(self, path: str, interface: str, connection: DBusConnection) -> None:
        DBusAddress.__init__(self, path, BUS_NAME, interface)
        self._connection = connection

    def send_and_get_reply(self, msg: Message) -> Any:
        try:
            resp_msg: Message = self._connection.send_and_get_reply(msg)
            if resp_msg.header.message_type == MessageType.error:
                raise DBusErrorResponse(resp_msg)
            return resp_msg.body
        except DBusErrorResponse as resp:
            if resp.name in (DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT):
                raise ItemNotFoundException('Item does not exist!') from resp
            elif resp.name in (DBUS_SERVICE_UNKNOWN, DBUS_EXEC_FAILED, DBUS_NO_REPLY):
                data = resp.data
                if isinstance(data, tuple):
                    data = data[0]
                raise SecretServiceNotAvailableException(data) from resp
            raise

    def call(self, method: str, signature: str, *body: Any) -> Any:
        msg = new_method_call(self, method, signature, body)
        return self.send_and_get_reply(msg)

    def get_property(self, name: str) -> Any:
        msg = Properties(self).get(name)
        (signature, value), = self.send_and_get_reply(msg)
        return value

    def set_property(self, name: str, signature: str, value: Any) -> None:
        msg = Properties(self).set(name, signature, value)
        self.send_and_get_reply(msg)