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
def _parse_vaulttext(b_vaulttext):
    b_vaulttext = _unhexlify(b_vaulttext)
    b_salt, b_crypted_hmac, b_ciphertext = b_vaulttext.split(b'\n', 2)
    b_salt = _unhexlify(b_salt)
    b_ciphertext = _unhexlify(b_ciphertext)
    return (b_ciphertext, b_salt, b_crypted_hmac)