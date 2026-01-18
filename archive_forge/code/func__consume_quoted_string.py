import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _consume_quoted_string(css, pos):
    """Return (unescaped_value, new_pos)."""
    error = None
    quote = css[pos]
    assert quote in ('"', "'")
    pos += 1
    chunks = []
    length = len(css)
    start_pos = pos
    while pos < length:
        c = css[pos]
        if c == quote:
            chunks.append(css[start_pos:pos])
            pos += 1
            break
        elif c == '\\':
            chunks.append(css[start_pos:pos])
            pos += 1
            if pos < length:
                if css[pos] == '\n':
                    pos += 1
                else:
                    c, pos = _consume_escape(css, pos)
                    chunks.append(c)
            start_pos = pos
        elif c == '\n':
            return (None, pos, ('bad-string', 'Bad string token'))
        else:
            pos += 1
    else:
        error = ('eof-in-string', 'EOF in string')
        chunks.append(css[start_pos:pos])
    return (''.join(chunks), pos, error)