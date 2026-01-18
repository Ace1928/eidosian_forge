from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class DHParameterNumbers:

    def __init__(self, p: int, g: int, q: typing.Optional[int]=None) -> None:
        if not isinstance(p, int) or not isinstance(g, int):
            raise TypeError('p and g must be integers')
        if q is not None and (not isinstance(q, int)):
            raise TypeError('q must be integer or None')
        if g < 2:
            raise ValueError('DH generator must be 2 or greater')
        if p.bit_length() < rust_openssl.dh.MIN_MODULUS_SIZE:
            raise ValueError(f'p (modulus) must be at least {rust_openssl.dh.MIN_MODULUS_SIZE}-bit')
        self._p = p
        self._g = g
        self._q = q

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHParameterNumbers):
            return NotImplemented
        return self._p == other._p and self._g == other._g and (self._q == other._q)

    def parameters(self, backend: typing.Any=None) -> DHParameters:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dh_parameter_numbers(self)

    @property
    def p(self) -> int:
        return self._p

    @property
    def g(self) -> int:
        return self._g

    @property
    def q(self) -> typing.Optional[int]:
        return self._q