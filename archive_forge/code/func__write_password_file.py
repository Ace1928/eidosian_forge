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
def _write_password_file(b_path, content):
    b_pathdir = os.path.dirname(b_path)
    makedirs_safe(b_pathdir, mode=448)
    with open(b_path, 'wb') as f:
        os.chmod(b_path, 384)
        b_content = to_bytes(content, errors='surrogate_or_strict') + b'\n'
        f.write(b_content)