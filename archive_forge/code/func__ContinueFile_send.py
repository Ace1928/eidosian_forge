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
def _ContinueFile_send(self):
    self._ContinueFile_write('HTTP/1.1 100 Continue\r\n\r\n'.encode('utf-8'))
    rfile = self._ContinueFile_rfile
    for attr in ('read', 'readline', 'readlines'):
        if hasattr(rfile, attr):
            setattr(self, attr, getattr(rfile, attr))