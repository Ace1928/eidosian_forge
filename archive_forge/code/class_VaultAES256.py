from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
class VaultAES256:
    """
    Vault implementation using AES-CTR with an HMAC-SHA256 authentication code.
    Keys are derived using PBKDF2
    """

    def __init__(self):
        if not HAS_CRYPTOGRAPHY:
            raise AnsibleError(NEED_CRYPTO_LIBRARY)

    @staticmethod
    def _create_key_cryptography(b_password, b_salt, key_length, iv_length):
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=2 * key_length + iv_length, salt=b_salt, iterations=10000, backend=CRYPTOGRAPHY_BACKEND)
        b_derivedkey = kdf.derive(b_password)
        return b_derivedkey

    @classmethod
    def _gen_key_initctr(cls, b_password, b_salt):
        key_length = 32
        if HAS_CRYPTOGRAPHY:
            iv_length = algorithms.AES.block_size // 8
            b_derivedkey = cls._create_key_cryptography(b_password, b_salt, key_length, iv_length)
            b_iv = b_derivedkey[key_length * 2:key_length * 2 + iv_length]
        else:
            raise AnsibleError(NEED_CRYPTO_LIBRARY + '(Detected in initctr)')
        b_key1 = b_derivedkey[:key_length]
        b_key2 = b_derivedkey[key_length:key_length * 2]
        return (b_key1, b_key2, b_iv)

    @staticmethod
    def _encrypt_cryptography(b_plaintext, b_key1, b_key2, b_iv):
        cipher = C_Cipher(algorithms.AES(b_key1), modes.CTR(b_iv), CRYPTOGRAPHY_BACKEND)
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        b_ciphertext = encryptor.update(padder.update(b_plaintext) + padder.finalize())
        b_ciphertext += encryptor.finalize()
        hmac = HMAC(b_key2, hashes.SHA256(), CRYPTOGRAPHY_BACKEND)
        hmac.update(b_ciphertext)
        b_hmac = hmac.finalize()
        return (to_bytes(hexlify(b_hmac), errors='surrogate_or_strict'), hexlify(b_ciphertext))

    @classmethod
    def _get_salt(cls):
        custom_salt = C.config.get_config_value('VAULT_ENCRYPT_SALT')
        if not custom_salt:
            custom_salt = os.urandom(32)
        return to_bytes(custom_salt)

    @classmethod
    def encrypt(cls, b_plaintext, secret, salt=None):
        if secret is None:
            raise AnsibleVaultError('The secret passed to encrypt() was None')
        if salt is None:
            b_salt = cls._get_salt()
        elif not salt:
            raise AnsibleVaultError('Empty or invalid salt passed to encrypt()')
        else:
            b_salt = to_bytes(salt)
        b_password = secret.bytes
        b_key1, b_key2, b_iv = cls._gen_key_initctr(b_password, b_salt)
        if HAS_CRYPTOGRAPHY:
            b_hmac, b_ciphertext = cls._encrypt_cryptography(b_plaintext, b_key1, b_key2, b_iv)
        else:
            raise AnsibleError(NEED_CRYPTO_LIBRARY + '(Detected in encrypt)')
        b_vaulttext = b'\n'.join([hexlify(b_salt), b_hmac, b_ciphertext])
        b_vaulttext = hexlify(b_vaulttext)
        return b_vaulttext

    @classmethod
    def _decrypt_cryptography(cls, b_ciphertext, b_crypted_hmac, b_key1, b_key2, b_iv):
        hmac = HMAC(b_key2, hashes.SHA256(), CRYPTOGRAPHY_BACKEND)
        hmac.update(b_ciphertext)
        try:
            hmac.verify(_unhexlify(b_crypted_hmac))
        except InvalidSignature as e:
            raise AnsibleVaultError('HMAC verification failed: %s' % e)
        cipher = C_Cipher(algorithms.AES(b_key1), modes.CTR(b_iv), CRYPTOGRAPHY_BACKEND)
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        b_plaintext = unpadder.update(decryptor.update(b_ciphertext) + decryptor.finalize()) + unpadder.finalize()
        return b_plaintext

    @staticmethod
    def _is_equal(b_a, b_b):
        """
        Comparing 2 byte arrays in constant time to avoid timing attacks.

        It would be nice if there were a library for this but hey.
        """
        if not (isinstance(b_a, binary_type) and isinstance(b_b, binary_type)):
            raise TypeError('_is_equal can only be used to compare two byte strings')
        if len(b_a) != len(b_b):
            return False
        result = 0
        for b_x, b_y in zip(b_a, b_b):
            result |= b_x ^ b_y
        return result == 0

    @classmethod
    def decrypt(cls, b_vaulttext, secret):
        b_ciphertext, b_salt, b_crypted_hmac = parse_vaulttext(b_vaulttext)
        b_password = secret.bytes
        b_key1, b_key2, b_iv = cls._gen_key_initctr(b_password, b_salt)
        if HAS_CRYPTOGRAPHY:
            b_plaintext = cls._decrypt_cryptography(b_ciphertext, b_crypted_hmac, b_key1, b_key2, b_iv)
        else:
            raise AnsibleError(NEED_CRYPTO_LIBRARY + '(Detected in decrypt)')
        return b_plaintext