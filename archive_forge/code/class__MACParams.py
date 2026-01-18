from __future__ import annotations
import binascii
import hmac
import struct
import types
import zlib
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Any, Callable, Dict, Tuple, Union
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing_extensions import Literal
from twisted import __version__ as twisted_version
from twisted.conch.ssh import _kex, address, keys
from twisted.conch.ssh.common import MP, NS, ffs, getMP, getNS
from twisted.internet import defer, protocol
from twisted.logger import Logger
from twisted.python import randbytes
from twisted.python.compat import iterbytes, networkString
class _MACParams(Tuple[_DigestMod, bytes, bytes, int]):
    """
    L{_MACParams} represents the parameters necessary to compute SSH MAC
    (Message Authenticate Codes).

    L{_MACParams} is a L{tuple} subclass to maintain compatibility with older
    versions of the code.  The elements of a L{_MACParams} are::

        0. The digest object used for the MAC
        1. The inner pad ("ipad") string
        2. The outer pad ("opad") string
        3. The size of the digest produced by the digest object

    L{_MACParams} is also an object lesson in why tuples are a bad type for
    public APIs.

    @ivar key: The HMAC key which will be used.
    """
    key: bytes