import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def _get_crypt(method: str, rc4_key: bytes, aes128_key: bytes, aes256_key: bytes) -> CryptBase:
    if method == '/AESV3':
        return CryptAES(aes256_key)
    if method == '/AESV2':
        return CryptAES(aes128_key)
    elif method == '/Identity':
        return CryptIdentity()
    else:
        return CryptRC4(rc4_key)