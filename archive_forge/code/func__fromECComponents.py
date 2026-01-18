from __future__ import annotations
import binascii
import struct
import unicodedata
import warnings
from base64 import b64encode, decodebytes, encodebytes
from hashlib import md5, sha256
from typing import Any
import bcrypt
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
from typing_extensions import Literal
from twisted.conch.ssh import common, sexpy
from twisted.conch.ssh.common import int_to_bytes
from twisted.python import randbytes
from twisted.python.compat import iterbytes, nativeString
from twisted.python.constants import NamedConstant, Names
from twisted.python.deprecate import _mutuallyExclusiveArguments
@classmethod
def _fromECComponents(cls, x, y, curve, privateValue=None):
    """
        Build a key from EC components.

        @param x: The affine x component of the public point used for verifying.
        @type x: L{int}

        @param y: The affine y component of the public point used for verifying.
        @type y: L{int}

        @param curve: NIST name of elliptic curve.
        @type curve: L{bytes}

        @param privateValue: The private value.
        @type privateValue: L{int}
        """
    publicNumbers = ec.EllipticCurvePublicNumbers(x=x, y=y, curve=_curveTable[curve])
    if privateValue is None:
        keyObject = publicNumbers.public_key(default_backend())
    else:
        privateNumbers = ec.EllipticCurvePrivateNumbers(private_value=privateValue, public_numbers=publicNumbers)
        keyObject = privateNumbers.private_key(default_backend())
    return cls(keyObject)