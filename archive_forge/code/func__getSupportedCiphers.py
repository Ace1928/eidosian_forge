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
def _getSupportedCiphers():
    """
    Build a list of ciphers that are supported by the backend in use.

    @return: a list of supported ciphers.
    @rtype: L{list} of L{str}
    """
    supportedCiphers = []
    cs = [b'aes256-ctr', b'aes256-cbc', b'aes192-ctr', b'aes192-cbc', b'aes128-ctr', b'aes128-cbc', b'cast128-ctr', b'cast128-cbc', b'blowfish-ctr', b'blowfish-cbc', b'3des-ctr', b'3des-cbc']
    for cipher in cs:
        algorithmClass, keySize, modeClass = SSHCiphers.cipherMap[cipher]
        try:
            Cipher(algorithmClass(b' ' * keySize), modeClass(b' ' * (algorithmClass.block_size // 8)), backend=default_backend()).encryptor()
        except UnsupportedAlgorithm:
            pass
        else:
            supportedCiphers.append(cipher)
    return supportedCiphers