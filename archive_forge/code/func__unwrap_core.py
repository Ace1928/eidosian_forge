from __future__ import annotations
import typing
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import ECB
from cryptography.hazmat.primitives.constant_time import bytes_eq
def _unwrap_core(wrapping_key: bytes, a: bytes, r: typing.List[bytes]) -> typing.Tuple[bytes, typing.List[bytes]]:
    decryptor = Cipher(AES(wrapping_key), ECB()).decryptor()
    n = len(r)
    for j in reversed(range(6)):
        for i in reversed(range(n)):
            atr = (int.from_bytes(a, byteorder='big') ^ n * j + i + 1).to_bytes(length=8, byteorder='big') + r[i]
            b = decryptor.update(atr)
            a = b[:8]
            r[i] = b[-8:]
    assert decryptor.finalize() == b''
    return (a, r)