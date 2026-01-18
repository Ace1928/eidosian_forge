import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _encode_reserved(text, maximal=True):
    """A very comprehensive percent encoding for encoding all
    delimiters. Used for arguments to DecodedURL, where a % means a
    percent sign, and not the character used by URLs for escaping
    bytes.
    """
    if maximal:
        bytestr = normalize('NFC', text).encode('utf8')
        return u''.join([_UNRESERVED_QUOTE_MAP[b] for b in bytestr])
    return u''.join([_UNRESERVED_QUOTE_MAP[t] if t in _UNRESERVED_CHARS else t for t in text])