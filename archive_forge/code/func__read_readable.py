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
def _read_readable(self, readable):
    if self.debuglevel > 0:
        print('reading a readable')
    encode = self._is_textIO(readable)
    if encode and self.debuglevel > 0:
        print('encoding file using iso-8859-1')
    while True:
        datablock = readable.read(self.blocksize)
        if not datablock:
            break
        if encode:
            datablock = datablock.encode('iso-8859-1')
        yield datablock