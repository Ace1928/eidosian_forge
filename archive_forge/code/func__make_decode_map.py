import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _make_decode_map(delims, allow_percent=False):
    ret = dict(_HEX_CHAR_MAP)
    if not allow_percent:
        delims = set(delims) | set([u'%'])
    for delim in delims:
        _hexord = '{0:02X}'.format(ord(delim)).encode('ascii')
        _hexord_lower = _hexord.lower()
        ret.pop(_hexord)
        if _hexord != _hexord_lower:
            ret.pop(_hexord_lower)
    return ret