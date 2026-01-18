import dataclasses
import socket
import ssl
import threading
import typing as t
class MessageEncryptor:
    """Message encryptor for LDAP client.

    Base object used by the LDAP client to encrypt and decrypt messages.
    """

    def wrap(self, data: bytes) -> bytes:
        raise NotImplementedError()

    def unwrap(self, data: bytes) -> t.Tuple[bytes, int]:
        raise NotImplementedError()