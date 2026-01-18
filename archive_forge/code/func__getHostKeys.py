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
def _getHostKeys(self, keyAlg):
    """
        Get the public and private host keys corresponding to the given
        public key signature algorithm.

        The factory stores public and private host keys by their key format,
        which is not quite the same as the key signature algorithm: for
        example, an ssh-rsa key can sign using any of the ssh-rsa,
        rsa-sha2-256, or rsa-sha2-512 algorithms.

        @type keyAlg: L{bytes}
        @param keyAlg: A public key signature algorithm name.

        @rtype: 2-L{tuple} of L{keys.Key}
        @return: The public and private host keys.

        @raises KeyError: if the factory does not have both a public and a
        private host key for this signature algorithm.
        """
    if keyAlg in {b'rsa-sha2-256', b'rsa-sha2-512'}:
        keyFormat = b'ssh-rsa'
    else:
        keyFormat = keyAlg
    return (self.factory.publicKeys[keyFormat], self.factory.privateKeys[keyFormat])