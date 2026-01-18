from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized, InvalidKey
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
class ConcatKDFHash(KeyDerivationFunction):

    def __init__(self, algorithm: hashes.HashAlgorithm, length: int, otherinfo: typing.Optional[bytes], backend: typing.Any=None):
        _common_args_checks(algorithm, length, otherinfo)
        self._algorithm = algorithm
        self._length = length
        self._otherinfo: bytes = otherinfo if otherinfo is not None else b''
        self._used = False

    def _hash(self) -> hashes.Hash:
        return hashes.Hash(self._algorithm)

    def derive(self, key_material: bytes) -> bytes:
        if self._used:
            raise AlreadyFinalized
        self._used = True
        return _concatkdf_derive(key_material, self._length, self._hash, self._otherinfo)

    def verify(self, key_material: bytes, expected_key: bytes) -> None:
        if not constant_time.bytes_eq(self.derive(key_material), expected_key):
            raise InvalidKey