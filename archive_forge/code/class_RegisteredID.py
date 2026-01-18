from __future__ import annotations
import abc
import ipaddress
import typing
from email.utils import parseaddr
from cryptography.x509.name import Name
from cryptography.x509.oid import ObjectIdentifier
class RegisteredID(GeneralName):

    def __init__(self, value: ObjectIdentifier) -> None:
        if not isinstance(value, ObjectIdentifier):
            raise TypeError('value must be an ObjectIdentifier')
        self._value = value

    @property
    def value(self) -> ObjectIdentifier:
        return self._value

    def __repr__(self) -> str:
        return f'<RegisteredID(value={self.value})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisteredID):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)