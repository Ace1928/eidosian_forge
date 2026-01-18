import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def _handle_no_ssl(self, req):
    if not req or req.sent_headers:
        return
    try:
        resp_sock = self.socket._sock
    except AttributeError:
        resp_sock = self.socket._socket
    self.wfile = StreamWriter(resp_sock, 'wb', self.wbufsize)
    msg = 'The client sent a plain HTTP request, but this server only speaks HTTPS on this port.'
    req.simple_response('400 Bad Request', msg)
    self.linger = True