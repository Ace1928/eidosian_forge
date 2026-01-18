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
def _ecdsa_key_type(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """Return SSH key_type and curve_name for private key."""
    curve = public_key.curve
    if curve.name not in _ECDSA_KEY_TYPE:
        raise ValueError(f'Unsupported curve for ssh private key: {curve.name!r}')
    return _ECDSA_KEY_TYPE[curve.name]