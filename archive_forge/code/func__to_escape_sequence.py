from __future__ import absolute_import
import re
import sys
def _to_escape_sequence(s):
    if s in '\n\r\t':
        return repr(s)[1:-1]
    elif s == '"':
        return '\\"'
    elif s == '\\':
        return '\\\\'
    else:
        return ''.join(['\\%03o' % ord(c) for c in s])