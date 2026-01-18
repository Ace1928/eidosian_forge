from Cryptodome.Util.py3compat import bchr, concat_buffers
from Cryptodome.Util._raw_api import (VoidPointer, SmartPointer,
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Hash.keccak import _raw_keccak_lib
def _bytepad(x, length):
    """Zero pad byte string as defined in NIST SP 800-185"""
    to_pad = concat_buffers(_left_encode(length), x)
    npad = (length - len(to_pad) % length) % length
    return to_pad + b'\x00' * npad