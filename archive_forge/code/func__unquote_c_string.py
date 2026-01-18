from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _unquote_c_string(s):
    """replace C-style escape sequences (
, ", etc.) with real chars."""

    def decode_match(match):
        return utf8_bytes_string(codecs.decode(match.group(0), 'unicode-escape'))
    if isinstance(s, bytes):
        return ESCAPE_SEQUENCE_BYTES_RE.sub(decode_match, s)
    else:
        return ESCAPE_SEQUENCE_RE.sub(decode_match, s)