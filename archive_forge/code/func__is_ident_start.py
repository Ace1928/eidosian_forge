import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _is_ident_start(css, pos):
    """Return True if the given position is the start of a CSS identifier."""
    if _is_name_start(css, pos):
        return True
    elif css[pos] == '-':
        pos += 1
        return pos < len(css) and (_is_name_start(css, pos) or css[pos] == '-') or (css.startswith('\\', pos) and (not css.startswith('\\\n', pos)))
    elif css[pos] == '\\':
        return not css.startswith('\\\n', pos)
    return False