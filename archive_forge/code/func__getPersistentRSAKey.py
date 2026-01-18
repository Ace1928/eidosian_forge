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
def _getPersistentRSAKey(location, keySize=4096):
    """
    This function returns a persistent L{Key}.

    The key is loaded from a PEM file in C{location}. If it does not exist, a
    key with the key size of C{keySize} is generated and saved.

    @param location: Where the key is stored.
    @type location: L{twisted.python.filepath.FilePath}

    @param keySize: The size of the key, if it needs to be generated.
    @type keySize: L{int}

    @returns: A persistent key.
    @rtype: L{Key}
    """
    location.parent().makedirs(ignoreExistingDirectory=True)
    if not location.exists():
        privateKey = rsa.generate_private_key(public_exponent=65537, key_size=keySize, backend=default_backend())
        pem = privateKey.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption())
        location.setContent(pem)
    with location.open('rb') as keyFile:
        privateKey = serialization.load_pem_private_key(keyFile.read(), password=None, backend=default_backend())
        return Key(privateKey)