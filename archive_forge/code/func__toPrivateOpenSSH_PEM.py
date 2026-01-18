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
def _toPrivateOpenSSH_PEM(self, passphrase=None):
    """
        Return a private OpenSSH key string, in the old PEM-based format.

        See _fromPrivateOpenSSH_PEM for the string format.

        @type passphrase: L{bytes} or L{None}
        @param passphrase: The passphrase to encrypt the key with, or L{None}
        if it is not encrypted.
        """
    if not passphrase:
        encryptor = serialization.NoEncryption()
    else:
        encryptor = serialization.BestAvailableEncryption(passphrase)
    if self.type() != 'Ed25519':
        return self._keyObject.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, encryptor)
    else:
        assert self.type() == 'Ed25519'
        raise ValueError('cannot serialize Ed25519 key to OpenSSH PEM format; use v1 instead')