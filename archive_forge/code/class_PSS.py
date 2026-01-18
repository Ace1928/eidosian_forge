from __future__ import annotations
import abc
import typing
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
from cryptography.hazmat.primitives.asymmetric import rsa
class PSS(AsymmetricPadding):
    MAX_LENGTH = _MaxLength()
    AUTO = _Auto()
    DIGEST_LENGTH = _DigestLength()
    name = 'EMSA-PSS'
    _salt_length: typing.Union[int, _MaxLength, _Auto, _DigestLength]

    def __init__(self, mgf: MGF, salt_length: typing.Union[int, _MaxLength, _Auto, _DigestLength]) -> None:
        self._mgf = mgf
        if not isinstance(salt_length, (int, _MaxLength, _Auto, _DigestLength)):
            raise TypeError('salt_length must be an integer, MAX_LENGTH, DIGEST_LENGTH, or AUTO')
        if isinstance(salt_length, int) and salt_length < 0:
            raise ValueError('salt_length must be zero or greater.')
        self._salt_length = salt_length