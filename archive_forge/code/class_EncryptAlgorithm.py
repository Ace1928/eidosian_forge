import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
class EncryptAlgorithm(tuple, Enum):
    RC4_40 = (1, 2, 40)
    RC4_128 = (2, 3, 128)
    AES_128 = (4, 4, 128)
    AES_256_R5 = (5, 5, 256)
    AES_256 = (5, 6, 256)