from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class GCM(ModeWithInitializationVector, ModeWithAuthenticationTag):
    name = 'GCM'
    _MAX_ENCRYPTED_BYTES = (2 ** 39 - 256) // 8
    _MAX_AAD_BYTES = 2 ** 64 // 8

    def __init__(self, initialization_vector: bytes, tag: typing.Optional[bytes]=None, min_tag_length: int=16):
        utils._check_byteslike('initialization_vector', initialization_vector)
        if len(initialization_vector) < 8 or len(initialization_vector) > 128:
            raise ValueError('initialization_vector must be between 8 and 128 bytes (64 and 1024 bits).')
        self._initialization_vector = initialization_vector
        if tag is not None:
            utils._check_bytes('tag', tag)
            if min_tag_length < 4:
                raise ValueError('min_tag_length must be >= 4')
            if len(tag) < min_tag_length:
                raise ValueError('Authentication tag must be {} bytes or longer.'.format(min_tag_length))
        self._tag = tag
        self._min_tag_length = min_tag_length

    @property
    def tag(self) -> typing.Optional[bytes]:
        return self._tag

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        _check_aes_key_length(self, algorithm)
        if not isinstance(algorithm, BlockCipherAlgorithm):
            raise UnsupportedAlgorithm('GCM requires a block cipher algorithm', _Reasons.UNSUPPORTED_CIPHER)
        block_size_bytes = algorithm.block_size // 8
        if self._tag is not None and len(self._tag) > block_size_bytes:
            raise ValueError('Authentication tag cannot be more than {} bytes.'.format(block_size_bytes))