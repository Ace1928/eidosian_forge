from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def _lookup_kformat(key_type: bytes):
    """Return valid format or throw error"""
    if not isinstance(key_type, bytes):
        key_type = memoryview(key_type).tobytes()
    if key_type in _KEY_FORMATS:
        return _KEY_FORMATS[key_type]
    raise UnsupportedAlgorithm(f'Unsupported key type: {key_type!r}')