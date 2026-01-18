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
def _get_u64(data: memoryview) -> typing.Tuple[int, memoryview]:
    """Uint64"""
    if len(data) < 8:
        raise ValueError('Invalid data')
    return (int.from_bytes(data[:8], byteorder='big'), data[8:])