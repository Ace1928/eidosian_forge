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
def _newKeys(self):
    """
        Called back by a subclass once a I{MSG_NEWKEYS} message has been
        received.  This indicates key exchange has completed and new encryption
        and compression parameters should be adopted.  Any messages which were
        queued during key exchange will also be flushed.
        """
    self._log.debug('NEW KEYS')
    self.currentEncryptions = self.nextEncryptions
    if self.outgoingCompressionType == b'zlib':
        self.outgoingCompression = zlib.compressobj(6)
    if self.incomingCompressionType == b'zlib':
        self.incomingCompression = zlib.decompressobj()
    self._keyExchangeState = self._KEY_EXCHANGE_NONE
    messages = self._blockedByKeyExchange
    self._blockedByKeyExchange = None
    for messageType, payload in messages:
        self.sendPacket(messageType, payload)