import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
def _make_crypt_filter(self, idnum: int, generation: int) -> CryptFilter:
    """
        Algorithm 1: Encryption of data using the RC4 or AES algorithms.

        a) Obtain the object number and generation number from the object
           identifier of the string or stream to be encrypted
           (see 7.3.10, "Indirect Objects"). If the string is a direct object,
           use the identifier of the indirect object containing it.
        b) For all strings and streams without crypt filter specifier; treating
           the object number and generation number as binary integers, extend
           the original n-byte encryption key to n + 5 bytes by appending the
           low-order 3 bytes of the object number and the low-order 2 bytes of
           the generation number in that order, low-order byte first.
           (n is 5 unless the value of V in the encryption dictionary is greater
           than 1, in which case n is the value of Length divided by 8.)
           If using the AES algorithm, extend the encryption key an additional
           4 bytes by adding the value “sAlT”, which corresponds to the
           hexadecimal values 0x73, 0x41, 0x6C, 0x54. (This addition is done for
           backward compatibility and is not intended to provide additional
           security.)
        c) Initialize the MD5 hash function and pass the result of step (b) as
           input to this function.
        d) Use the first (n + 5) bytes, up to a maximum of 16, of the output
           from the MD5 hash as the key for the RC4 or AES symmetric key
           algorithms, along with the string or stream data to be encrypted.
           If using the AES algorithm, the Cipher Block Chaining (CBC) mode,
           which requires an initialization vector, is used. The block size
           parameter is set to 16 bytes, and the initialization vector is a
           16-byte random number that is stored as the first 16 bytes of the
           encrypted stream or string.

        Algorithm 3.1a Encryption of data using the AES algorithm
        1. Use the 32-byte file encryption key for the AES-256 symmetric key
           algorithm, along with the string or stream data to be encrypted.
           Use the AES algorithm in Cipher Block Chaining (CBC) mode, which
           requires an initialization vector. The block size parameter is set to
           16 bytes, and the initialization vector is a 16-byte random number
           that is stored as the first 16 bytes of the encrypted stream or string.
           The output is the encrypted data to be stored in the PDF file.
        """
    pack1 = struct.pack('<i', idnum)[:3]
    pack2 = struct.pack('<i', generation)[:2]
    assert self._key
    key = self._key
    n = 5 if self.V == 1 else self.Length // 8
    key_data = key[:n] + pack1 + pack2
    key_hash = hashlib.md5(key_data)
    rc4_key = key_hash.digest()[:min(n + 5, 16)]
    key_hash.update(b'sAlT')
    aes128_key = key_hash.digest()[:min(n + 5, 16)]
    aes256_key = key
    stm_crypt = self._get_crypt(self.StmF, rc4_key, aes128_key, aes256_key)
    str_crypt = self._get_crypt(self.StrF, rc4_key, aes128_key, aes256_key)
    ef_crypt = self._get_crypt(self.EFF, rc4_key, aes128_key, aes256_key)
    return CryptFilter(stm_crypt, str_crypt, ef_crypt)