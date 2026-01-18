from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def DebugFormatString(self, value):

    def escape(c):
        o = ord(c)
        if o == 10:
            return '\\n'
        if o == 39:
            return "\\'"
        if o == 34:
            return '\\"'
        if o == 92:
            return '\\\\'
        if o >= 127 or o < 32:
            return '\\%03o' % o
        return c
    return '"' + ''.join((escape(c) for c in value)) + '"'