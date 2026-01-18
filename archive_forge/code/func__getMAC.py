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
def _getMAC(self, mac: bytes, key: bytes) -> tuple[None, Literal[b''], Literal[b''], Literal[0]] | _MACParams:
    """
        Gets a 4-tuple representing the message authentication code.
        (<hash module>, <inner hash value>, <outer hash value>,
        <digest size>)

        @type mac: L{bytes}
        @param mac: a key mapping into macMap

        @type key: L{bytes}
        @param key: the MAC key.

        @rtype: L{bytes}
        @return: The MAC components.
        """
    mod = self.macMap[mac]
    if not mod:
        return (None, b'', b'', 0)
    hashObject = mod()
    digestSize = hashObject.digest_size
    blockSize = hashObject.block_size
    key = key[:digestSize] + b'\x00' * (blockSize - digestSize)
    i = key.translate(hmac.trans_36)
    o = key.translate(hmac.trans_5C)
    result = _MACParams((mod, i, o, digestSize))
    result.key = key
    return result