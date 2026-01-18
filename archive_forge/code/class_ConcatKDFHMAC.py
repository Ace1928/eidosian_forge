from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized, InvalidKey
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
class ConcatKDFHMAC(KeyDerivationFunction):

    def __init__(self, algorithm: hashes.HashAlgorithm, length: int, salt: typing.Optional[bytes], otherinfo: typing.Optional[bytes], backend: typing.Any=None):
        _common_args_checks(algorithm, length, otherinfo)
        self._algorithm = algorithm
        self._length = length
        self._otherinfo: bytes = otherinfo if otherinfo is not None else b''
        if algorithm.block_size is None:
            raise TypeError(f'{algorithm.name} is unsupported for ConcatKDF')
        if salt is None:
            salt = b'\x00' * algorithm.block_size
        else:
            utils._check_bytes('salt', salt)
        self._salt = salt
        self._used = False

    def _hmac(self) -> hmac.HMAC:
        return hmac.HMAC(self._salt, self._algorithm)

    def derive(self, key_material: bytes) -> bytes:
        if self._used:
            raise AlreadyFinalized
        self._used = True
        return _concatkdf_derive(key_material, self._length, self._hmac, self._otherinfo)

    def verify(self, key_material: bytes, expected_key: bytes) -> None:
        if not constant_time.bytes_eq(self.derive(key_material), expected_key):
            raise InvalidKey