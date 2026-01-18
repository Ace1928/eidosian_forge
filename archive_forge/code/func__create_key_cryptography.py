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
def _create_key_cryptography(b_password, b_salt, key_length, iv_length):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=2 * key_length + iv_length, salt=b_salt, iterations=10000, backend=CRYPTOGRAPHY_BACKEND)
    b_derivedkey = kdf.derive(b_password)
    return b_derivedkey