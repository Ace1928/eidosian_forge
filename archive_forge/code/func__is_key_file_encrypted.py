from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
def _is_key_file_encrypted(key_file):
    """Detects if a key file is encrypted or not."""
    with open(key_file, 'r') as f:
        for line in f:
            if 'ENCRYPTED' in line:
                return True
    return False