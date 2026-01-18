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
def _fromDSAComponents(cls, y, p, q, g, x=None):
    """
        Build a key from DSA numerical components.

        @type y: L{int}
        @param y: The 'y' DSA variable.

        @type p: L{int}
        @param p: The 'p' DSA variable.

        @type q: L{int}
        @param q: The 'q' DSA variable.

        @type g: L{int}
        @param g: The 'g' DSA variable.

        @type x: L{int} or L{None}
        @param x: The 'x' DSA variable (optional for a public key)

        @rtype: L{Key}
        @return: A DSA key constructed from the values as given.
        """
    publicNumbers = dsa.DSAPublicNumbers(y=y, parameter_numbers=dsa.DSAParameterNumbers(p=p, q=q, g=g))
    if x is None:
        keyObject = publicNumbers.public_key(default_backend())
    else:
        privateNumbers = dsa.DSAPrivateNumbers(x=x, public_numbers=publicNumbers)
        keyObject = privateNumbers.private_key(default_backend())
    return cls(keyObject)