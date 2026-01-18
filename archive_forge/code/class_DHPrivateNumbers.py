from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class DHPrivateNumbers:

    def __init__(self, x: int, public_numbers: DHPublicNumbers) -> None:
        if not isinstance(x, int):
            raise TypeError('x must be an integer.')
        if not isinstance(public_numbers, DHPublicNumbers):
            raise TypeError('public_numbers must be an instance of DHPublicNumbers.')
        self._x = x
        self._public_numbers = public_numbers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHPrivateNumbers):
            return NotImplemented
        return self._x == other._x and self._public_numbers == other._public_numbers

    def private_key(self, backend: typing.Any=None) -> DHPrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dh_private_numbers(self)

    @property
    def public_numbers(self) -> DHPublicNumbers:
        return self._public_numbers

    @property
    def x(self) -> int:
        return self._x