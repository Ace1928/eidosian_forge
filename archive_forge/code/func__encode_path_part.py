import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _encode_path_part(text, maximal=True):
    """Percent-encode a single segment of a URL path."""
    if maximal:
        bytestr = normalize('NFC', text).encode('utf8')
        return u''.join([_PATH_PART_QUOTE_MAP[b] for b in bytestr])
    return u''.join([_PATH_PART_QUOTE_MAP[t] if t in _PATH_DELIMS else t for t in text])