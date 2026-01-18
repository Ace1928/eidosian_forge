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
def _generateECPrivateKey(self):
    """
        Generate an private key for ECDH key exchange.

        @rtype: The appropriate private key type matching C{self.kexAlg}:
            L{ec.EllipticCurvePrivateKey} for C{ecdh-sha2-nistp*}, or
            L{x25519.X25519PrivateKey} for C{curve25519-sha256}.
        @return: The generated private key.
        """
    if self.kexAlg.startswith(b'ecdh-sha2-nistp'):
        try:
            curve = keys._curveTable[b'ecdsa' + self.kexAlg[4:]]
        except KeyError:
            raise UnsupportedAlgorithm('unused-key')
        return ec.generate_private_key(curve, default_backend())
    elif self.kexAlg in (b'curve25519-sha256', b'curve25519-sha256@libssh.org'):
        return x25519.X25519PrivateKey.generate()
    else:
        raise UnsupportedAlgorithm('Cannot generate elliptic curve private key for {!r}'.format(self.kexAlg))