from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
class PKCS7:

    def __init__(self, block_size: int):
        _byte_padding_check(block_size)
        self.block_size = block_size

    def padder(self) -> PaddingContext:
        return _PKCS7PaddingContext(self.block_size)

    def unpadder(self) -> PaddingContext:
        return _PKCS7UnpaddingContext(self.block_size)