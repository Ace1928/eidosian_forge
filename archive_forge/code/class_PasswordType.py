import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
class PasswordType(IntEnum):
    NOT_DECRYPTED = 0
    USER_PASSWORD = 1
    OWNER_PASSWORD = 2