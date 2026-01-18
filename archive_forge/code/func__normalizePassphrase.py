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
def _normalizePassphrase(passphrase):
    """
    Normalize a passphrase, which may be Unicode.

    If the passphrase is Unicode, this follows the requirements of U{NIST
    800-63B, section
    5.1.1.2<https://pages.nist.gov/800-63-3/sp800-63b.html#memsecretver>}
    for Unicode characters in memorized secrets: it applies the
    Normalization Process for Stabilized Strings using NFKC normalization.
    The passphrase is then encoded using UTF-8.

    @type passphrase: L{bytes} or L{unicode} or L{None}
    @param passphrase: The passphrase to normalize.

    @return: The normalized passphrase, if any.
    @rtype: L{bytes} or L{None}
    @raises PassphraseNormalizationError: if the passphrase is Unicode and
    cannot be normalized using the available Unicode character database.
    """
    if isinstance(passphrase, str):
        if any((unicodedata.category(c) == 'Cn' for c in passphrase)):
            raise PassphraseNormalizationError()
        return unicodedata.normalize('NFKC', passphrase).encode('UTF-8')
    else:
        return passphrase