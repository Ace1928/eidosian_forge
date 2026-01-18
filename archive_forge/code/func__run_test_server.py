import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
def _run_test_server(*, address, use_ssl=False, server_cls, server_ssl_cls):

    def app(environ, start_response):
        status = '200 OK'
        headers = [('Content-type', 'text/plain')]
        start_response(status, headers)
        return [b'Test message']
    server_class = server_ssl_cls if use_ssl else server_cls
    httpd = server_class(address, SilentWSGIRequestHandler)
    httpd.set_app(app)
    httpd.address = httpd.server_address
    server_thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.05))
    server_thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()