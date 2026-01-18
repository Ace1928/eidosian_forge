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
def _validate_host(self, host):
    """Validate a host so it doesn't contain control characters."""
    match = _contains_disallowed_url_pchar_re.search(host)
    if match:
        raise InvalidURL(f"URL can't contain control characters. {host!r} (found at least {match.group()!r})")