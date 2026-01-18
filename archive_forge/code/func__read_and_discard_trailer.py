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
def _read_and_discard_trailer(self):
    while True:
        line = self.fp.readline(_MAXLINE + 1)
        if len(line) > _MAXLINE:
            raise LineTooLong('trailer line')
        if not line:
            break
        if line in (b'\r\n', b'\n', b''):
            break