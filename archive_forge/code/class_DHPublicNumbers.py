from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class DHPublicNumbers:

    def __init__(self, y: int, parameter_numbers: DHParameterNumbers) -> None:
        if not isinstance(y, int):
            raise TypeError('y must be an integer.')
        if not isinstance(parameter_numbers, DHParameterNumbers):
            raise TypeError('parameters must be an instance of DHParameterNumbers.')
        self._y = y
        self._parameter_numbers = parameter_numbers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHPublicNumbers):
            return NotImplemented
        return self._y == other._y and self._parameter_numbers == other._parameter_numbers

    def public_key(self, backend: typing.Any=None) -> DHPublicKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dh_public_numbers(self)

    @property
    def y(self) -> int:
        return self._y

    @property
    def parameter_numbers(self) -> DHParameterNumbers:
        return self._parameter_numbers