import base64
import sys
from cryptography import fernet
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.primitives import padding
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from heat.common import exception
from heat.common.i18n import _
class SymmetricCrypto(object):
    """Symmetric Key Crypto object.

    This class creates a Symmetric Key Crypto object that can be used
    to decrypt arbitrary data.

    Note: This is a reimplementation of the decryption algorithm from
    oslo-incubator, and is provided for backward compatibility. Once we have a
    DB migration script available to re-encrypt using new encryption method as
    part of upgrade, this can be removed.

    :param enctype: Encryption Cipher name (default: AES)
    """

    def __init__(self, enctype='AES'):
        self.algo = algorithms.AES

    def decrypt(self, key, msg, b64decode=True):
        """Decrypts the provided ciphertext.

        The ciphertext can be optionally base64 encoded.

        Uses AES-128-CBC with an IV by default.

        :param key: The Encryption key.
        :param msg: the ciphetext, the first block is the IV

        :returns: the plaintext message, after padding is removed.
        """
        key = str.encode(get_valid_encryption_key(key))
        if b64decode:
            msg = base64.b64decode(msg)
        algo = self.algo(key)
        block_size_bytes = algo.block_size // 8
        iv = msg[:block_size_bytes]
        backend = backends.default_backend()
        cipher = Cipher(algo, modes.CBC(iv), backend=backend)
        decryptor = cipher.decryptor()
        padded = decryptor.update(msg[block_size_bytes:]) + decryptor.finalize()
        unpadder = padding.ANSIX923(algo.block_size).unpadder()
        plain = unpadder.update(padded) + unpadder.finalize()
        return plain[:-1]