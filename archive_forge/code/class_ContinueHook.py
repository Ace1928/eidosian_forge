import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
class ContinueHook(object):
    """
    When a client request includes a 'Expect: 100-continue' header, then
    it is the responsibility of the server to send 100 Continue when it
    is ready for the content body.  This allows authentication, access
    levels, and other exceptions to be detected *before* bandwith is
    spent on the request body.

    This is a rfile wrapper that implements this functionality by
    sending 100 Continue to the client immediately after the user
    requests the content via a read() operation on the rfile stream.
    After this response is sent, it becomes a pass-through object.
    """

    def __init__(self, rfile, write):
        self._ContinueFile_rfile = rfile
        self._ContinueFile_write = write
        for attr in ('close', 'closed', 'fileno', 'flush', 'mode', 'bufsize', 'softspace'):
            if hasattr(rfile, attr):
                setattr(self, attr, getattr(rfile, attr))
        for attr in ('read', 'readline', 'readlines'):
            if hasattr(rfile, attr):
                setattr(self, attr, getattr(self, '_ContinueFile_' + attr))

    def _ContinueFile_send(self):
        self._ContinueFile_write('HTTP/1.1 100 Continue\r\n\r\n'.encode('utf-8'))
        rfile = self._ContinueFile_rfile
        for attr in ('read', 'readline', 'readlines'):
            if hasattr(rfile, attr):
                setattr(self, attr, getattr(rfile, attr))

    def _ContinueFile_read(self, size=-1):
        self._ContinueFile_send()
        return self._ContinueFile_rfile.read(size)

    def _ContinueFile_readline(self, size=-1):
        self._ContinueFile_send()
        return self._ContinueFile_rfile.readline(size)

    def _ContinueFile_readlines(self, sizehint=0):
        self._ContinueFile_send()
        return self._ContinueFile_rfile.readlines(sizehint)