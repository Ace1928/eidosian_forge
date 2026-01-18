from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized, InvalidKey
from cryptography.hazmat.primitives import constant_time, hashes
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
def _int_to_u32be(n: int) -> bytes:
    return n.to_bytes(length=4, byteorder='big')