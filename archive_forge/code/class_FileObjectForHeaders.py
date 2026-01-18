import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
class FileObjectForHeaders:

    def __init__(self, fp):
        self.fp = fp
        self.total_header_size = 0

    def readline(self, size=-1):
        sz = size
        if size < 0:
            sz = MAX_HEADER_LINE
        rv = self.fp.readline(sz)
        if len(rv) >= MAX_HEADER_LINE:
            raise HeaderLineTooLong()
        self.total_header_size += len(rv)
        if self.total_header_size > MAX_TOTAL_HEADER_SIZE:
            raise HeadersTooLarge()
        return rv