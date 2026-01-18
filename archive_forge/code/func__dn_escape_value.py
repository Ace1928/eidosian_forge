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
def _dn_escape_value(value):
    """
    Escape Distinguished Name's attribute value.
    """
    value = value.replace(u'\\', u'\\\\')
    for ch in [u',', u'+', u'<', u'>', u';', u'"']:
        value = value.replace(ch, u'\\%s' % ch)
    value = value.replace(u'\x00', u'\\00')
    if value.startswith((u' ', u'#')):
        value = u'\\%s' % value[0] + value[1:]
    if value.endswith(u' '):
        value = value[:-1] + u'\\ '
    return value