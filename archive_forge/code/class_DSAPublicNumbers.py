from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class DSAPublicNumbers:

    def __init__(self, y: int, parameter_numbers: DSAParameterNumbers):
        if not isinstance(y, int):
            raise TypeError('DSAPublicNumbers y argument must be an integer.')
        if not isinstance(parameter_numbers, DSAParameterNumbers):
            raise TypeError('parameter_numbers must be a DSAParameterNumbers instance.')
        self._y = y
        self._parameter_numbers = parameter_numbers

    @property
    def y(self) -> int:
        return self._y

    @property
    def parameter_numbers(self) -> DSAParameterNumbers:
        return self._parameter_numbers

    def public_key(self, backend: typing.Any=None) -> DSAPublicKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dsa_public_numbers(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DSAPublicNumbers):
            return NotImplemented
        return self.y == other.y and self.parameter_numbers == other.parameter_numbers

    def __repr__(self) -> str:
        return '<DSAPublicNumbers(y={self.y}, parameter_numbers={self.parameter_numbers})>'.format(self=self)