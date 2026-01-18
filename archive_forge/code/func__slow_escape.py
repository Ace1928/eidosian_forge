import codecs
import io
import locale
import os
import sys
import unicodedata
from io import StringIO, BytesIO
def _slow_escape(text):
    """Escape unicode ``text`` leaving printable characters unmodified

    The behaviour emulates the Python 3 implementation of repr, see
    unicode_repr in unicodeobject.c and isprintable definition.

    Because this iterates over the input a codepoint at a time, it's slow, and
    does not handle astral characters correctly on Python builds with 16 bit
    rather than 32 bit unicode type.
    """
    output = []
    for c in text:
        o = ord(c)
        if o < 256:
            if o < 32 or 126 < o < 161:
                output.append(c.encode('unicode-escape'))
            elif o == 92:
                output.append('\\\\')
            else:
                output.append(c)
        elif unicodedata.category(c)[0] in 'CZ':
            output.append(c.encode('unicode-escape'))
        else:
            output.append(c)
    return ''.join(output)