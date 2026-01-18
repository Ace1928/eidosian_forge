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
def _read_headers(fp):
    """Reads potential header lines into a list from a file pointer.

    Length of line is limited by _MAXLINE, and number of
    headers is limited by _MAXHEADERS.
    """
    headers = []
    while True:
        line = fp.readline(_MAXLINE + 1)
        if len(line) > _MAXLINE:
            raise LineTooLong('header line')
        headers.append(line)
        if len(headers) > _MAXHEADERS:
            raise HTTPException('got more than %d headers' % _MAXHEADERS)
        if line in (b'\r\n', b'\n', b''):
            break
    return headers