from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def _compare_public_keys(key1, key2, clazz):
    a = isinstance(key1, clazz)
    b = isinstance(key2, clazz)
    if not (a or b):
        return None
    if not a or not b:
        return False
    a = key1.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    b = key2.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return a == b