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
@staticmethod
def _get_content_length(body, method):
    """Get the content-length based on the body.

        If the body is None, we set Content-Length: 0 for methods that expect
        a body (RFC 7230, Section 3.3.2). We also set the Content-Length for
        any method if the body is a str or bytes-like object and not a file.
        """
    if body is None:
        if method.upper() in _METHODS_EXPECTING_BODY:
            return 0
        else:
            return None
    if hasattr(body, 'read'):
        return None
    try:
        mv = memoryview(body)
        return mv.nbytes
    except TypeError:
        pass
    if isinstance(body, str):
        return len(body)
    return None