from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
class _ANSIX923UnpaddingContext(PaddingContext):
    _buffer: typing.Optional[bytes]

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._buffer = b''

    def update(self, data: bytes) -> bytes:
        self._buffer, result = _byte_unpadding_update(self._buffer, data, self.block_size)
        return result

    def finalize(self) -> bytes:
        result = _byte_unpadding_check(self._buffer, self.block_size, check_ansix923_padding)
        self._buffer = None
        return result