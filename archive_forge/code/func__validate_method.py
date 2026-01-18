import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _validate_method(self, method):
    """Validate a method name for putrequest."""
    match = _contains_disallowed_method_pchar_re.search(method)
    if match:
        raise ValueError(f"method can't contain control characters. {method!r} (found at least {match.group()!r})")