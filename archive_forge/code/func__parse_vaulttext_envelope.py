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
def _parse_vaulttext_envelope(b_vaulttext_envelope, default_vault_id=None):
    b_tmpdata = b_vaulttext_envelope.splitlines()
    b_tmpheader = b_tmpdata[0].strip().split(b';')
    b_version = b_tmpheader[1].strip()
    cipher_name = to_text(b_tmpheader[2].strip())
    vault_id = default_vault_id
    if len(b_tmpheader) >= 4:
        vault_id = to_text(b_tmpheader[3].strip())
    b_ciphertext = b''.join(b_tmpdata[1:])
    return (b_ciphertext, b_version, cipher_name, vault_id)