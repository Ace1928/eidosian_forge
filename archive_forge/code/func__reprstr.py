from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def _reprstr(s):
    s = repr(s)
    if isinstance(s, str) and s.startswith("u'"):
        return s[2:-1]
    return s[1:-1]