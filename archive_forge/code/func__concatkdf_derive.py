from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized, InvalidKey
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
def _concatkdf_derive(key_material: bytes, length: int, auxfn: typing.Callable[[], hashes.HashContext], otherinfo: bytes) -> bytes:
    utils._check_byteslike('key_material', key_material)
    output = [b'']
    outlen = 0
    counter = 1
    while length > outlen:
        h = auxfn()
        h.update(_int_to_u32be(counter))
        h.update(key_material)
        h.update(otherinfo)
        output.append(h.finalize())
        outlen += len(output[-1])
        counter += 1
    return b''.join(output)[:length]