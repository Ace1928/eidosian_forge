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
def _adjust_idn_url(value, idn_rewrite):
    url = urlparse(value)
    host = _adjust_idn(url.hostname, idn_rewrite)
    if url.username is not None and url.password is not None:
        host = u'{0}:{1}@{2}'.format(url.username, url.password, host)
    elif url.username is not None:
        host = u'{0}@{1}'.format(url.username, host)
    if url.port is not None:
        host = u'{0}:{1}'.format(host, url.port)
    return urlunparse(ParseResult(scheme=url.scheme, netloc=host, path=url.path, params=url.params, query=url.query, fragment=url.fragment))