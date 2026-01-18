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
def _fromString_BLOB(cls, blob):
    """
        Return a public key object corresponding to this public key blob.
        The format of a RSA public key blob is::
            string 'ssh-rsa'
            integer e
            integer n

        The format of a DSA public key blob is::
            string 'ssh-dss'
            integer p
            integer q
            integer g
            integer y

        The format of ECDSA-SHA2-* public key blob is::
            string 'ecdsa-sha2-[identifier]'
            integer x
            integer y

            identifier is the standard NIST curve name.

        The format of an Ed25519 public key blob is::
            string 'ssh-ed25519'
            string a

        @type blob: L{bytes}
        @param blob: The key data.

        @return: A new key.
        @rtype: L{twisted.conch.ssh.keys.Key}
        @raises BadKeyError: if the key type (the first string) is unknown.
        """
    keyType, rest = common.getNS(blob)
    if keyType == b'ssh-rsa':
        e, n, rest = common.getMP(rest, 2)
        return cls(rsa.RSAPublicNumbers(e, n).public_key(default_backend()))
    elif keyType == b'ssh-dss':
        p, q, g, y, rest = common.getMP(rest, 4)
        return cls(dsa.DSAPublicNumbers(y=y, parameter_numbers=dsa.DSAParameterNumbers(p=p, q=q, g=g)).public_key(default_backend()))
    elif keyType in _curveTable:
        return cls(ec.EllipticCurvePublicKey.from_encoded_point(_curveTable[keyType], common.getNS(rest, 2)[1]))
    elif keyType == b'ssh-ed25519':
        a, rest = common.getNS(rest)
        return cls._fromEd25519Components(a)
    else:
        raise BadKeyError(f'unknown blob type: {keyType}')