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
class SecureHTTPServer(HTTPServer):

    def __init__(self, server_address, RequestHandlerClass, ssl_context=None, request_queue_size=None):
        assert not ssl_context, 'pyOpenSSL not installed'
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
        if request_queue_size:
            self.socket.listen(request_queue_size)