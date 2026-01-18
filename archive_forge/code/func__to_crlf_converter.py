import re
import sys
from ..errors import BzrError
from ..filters import ContentFilter
def _to_crlf_converter(chunks, context=None):
    """A content file that converts lf to crlf."""
    content = b''.join(chunks)
    if b'\x00' in content:
        return [content]
    else:
        return [_UNIX_NL_RE.sub(b'\r\n', content)]