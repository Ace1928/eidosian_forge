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
def _fromRSAComponents(cls, n, e, d=None, p=None, q=None, u=None):
    """
        Build a key from RSA numerical components.

        @type n: L{int}
        @param n: The 'n' RSA variable.

        @type e: L{int}
        @param e: The 'e' RSA variable.

        @type d: L{int} or L{None}
        @param d: The 'd' RSA variable (optional for a public key).

        @type p: L{int} or L{None}
        @param p: The 'p' RSA variable (optional for a public key).

        @type q: L{int} or L{None}
        @param q: The 'q' RSA variable (optional for a public key).

        @type u: L{int} or L{None}
        @param u: The 'u' RSA variable. Ignored, as its value is determined by
        p and q.

        @rtype: L{Key}
        @return: An RSA key constructed from the values as given.
        """
    publicNumbers = rsa.RSAPublicNumbers(e=e, n=n)
    if d is None:
        keyObject = publicNumbers.public_key(default_backend())
    else:
        privateNumbers = rsa.RSAPrivateNumbers(p=p, q=q, d=d, dmp1=rsa.rsa_crt_dmp1(d, p), dmq1=rsa.rsa_crt_dmq1(d, q), iqmp=rsa.rsa_crt_iqmp(p, q), public_numbers=publicNumbers)
        keyObject = privateNumbers.private_key(default_backend())
    return cls(keyObject)