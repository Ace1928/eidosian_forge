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