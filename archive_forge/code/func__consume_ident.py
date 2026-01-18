import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _consume_ident(css, pos):
    """Return (unescaped_value, new_pos).

    Assumes pos starts at a valid identifier. See :func:`_is_ident_start`.

    """
    chunks = []
    length = len(css)
    start_pos = pos
    while pos < length:
        c = css[pos]
        if c in 'abcdefghijklmnopqrstuvwxyz-_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' or ord(c) > 127:
            pos += 1
        elif c == '\\' and (not css.startswith('\\\n', pos)):
            chunks.append(css[start_pos:pos])
            c, pos = _consume_escape(css, pos + 1)
            chunks.append(c)
            start_pos = pos
        else:
            break
    chunks.append(css[start_pos:pos])
    return (''.join(chunks), pos)