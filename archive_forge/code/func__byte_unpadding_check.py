from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
def _byte_unpadding_check(buffer_: typing.Optional[bytes], block_size: int, checkfn: typing.Callable[[bytes], int]) -> bytes:
    if buffer_ is None:
        raise AlreadyFinalized('Context was already finalized.')
    if len(buffer_) != block_size // 8:
        raise ValueError('Invalid padding bytes.')
    valid = checkfn(buffer_)
    if not valid:
        raise ValueError('Invalid padding bytes.')
    pad_size = buffer_[-1]
    return buffer_[:-pad_size]