from __future__ import (absolute_import, division, print_function)
import os
import string
import time
import hashlib
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import string_types
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
from ansible.utils.encrypt import BaseHash, do_encrypt, random_password, random_salt
from ansible.utils.path import makedirs_safe
def _read_password_file(b_path):
    """Read the contents of a password file and return it
    :arg b_path: A byte string containing the path to the password file
    :returns: a text string containing the contents of the password file or
        None if no password file was present.
    """
    content = None
    if os.path.exists(b_path):
        with open(b_path, 'rb') as f:
            b_content = f.read().rstrip()
        content = to_text(b_content, errors='surrogate_or_strict')
    return content