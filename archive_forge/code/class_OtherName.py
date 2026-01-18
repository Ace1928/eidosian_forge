from __future__ import annotations
import abc
import ipaddress
import typing
from email.utils import parseaddr
from cryptography.x509.name import Name
from cryptography.x509.oid import ObjectIdentifier
class OtherName(GeneralName):

    def __init__(self, type_id: ObjectIdentifier, value: bytes) -> None:
        if not isinstance(type_id, ObjectIdentifier):
            raise TypeError('type_id must be an ObjectIdentifier')
        if not isinstance(value, bytes):
            raise TypeError('value must be a binary string')
        self._type_id = type_id
        self._value = value

    @property
    def type_id(self) -> ObjectIdentifier:
        return self._type_id

    @property
    def value(self) -> bytes:
        return self._value

    def __repr__(self) -> str:
        return '<OtherName(type_id={}, value={!r})>'.format(self.type_id, self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OtherName):
            return NotImplemented
        return self.type_id == other.type_id and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.type_id, self.value))