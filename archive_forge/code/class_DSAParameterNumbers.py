from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class DSAParameterNumbers:

    def __init__(self, p: int, q: int, g: int):
        if not isinstance(p, int) or not isinstance(q, int) or (not isinstance(g, int)):
            raise TypeError('DSAParameterNumbers p, q, and g arguments must be integers.')
        self._p = p
        self._q = q
        self._g = g

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def g(self) -> int:
        return self._g

    def parameters(self, backend: typing.Any=None) -> DSAParameters:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dsa_parameter_numbers(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DSAParameterNumbers):
            return NotImplemented
        return self.p == other.p and self.q == other.q and (self.g == other.g)

    def __repr__(self) -> str:
        return '<DSAParameterNumbers(p={self.p}, q={self.q}, g={self.g})>'.format(self=self)