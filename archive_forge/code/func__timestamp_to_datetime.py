from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@staticmethod
def _timestamp_to_datetime(timestamp):
    if timestamp == 0:
        result = _ALWAYS
    elif timestamp == 18446744073709551615:
        result = _FOREVER
    else:
        try:
            result = datetime.utcfromtimestamp(timestamp)
        except OverflowError as e:
            raise ValueError
    return result