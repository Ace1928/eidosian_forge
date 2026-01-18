import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
class ConnectionMarker(object):
    """
    Marks an HTTP(S)Connection's socket after a request was made.

    Helps a test server understand when a client finished a request,
    without implementing a complete HTTP server.
    """
    MARK_FORMAT = b'$#MARK%04x*!'

    @classmethod
    @contextmanager
    def mark(cls, monkeypatch):
        """
        Mark connections under in that context.
        """
        orig_request = HTTPConnection.request
        orig_request_chunked = HTTPConnection.request_chunked

        def call_and_mark(target):

            def part(self, *args, **kwargs):
                result = target(self, *args, **kwargs)
                self.sock.sendall(cls._get_socket_mark(self.sock, False))
                return result
            return part
        with monkeypatch.context() as m:
            m.setattr(HTTPConnection, 'request', call_and_mark(orig_request))
            m.setattr(HTTPConnection, 'request_chunked', call_and_mark(orig_request_chunked))
            yield

    @classmethod
    def consume_request(cls, sock, chunks=65536):
        """
        Consume a socket until after the HTTP request is sent.
        """
        consumed = bytearray()
        mark = cls._get_socket_mark(sock, True)
        while True:
            b = sock.recv(chunks)
            if not b:
                break
            consumed += b
            if consumed.endswith(mark):
                break
        return consumed

    @classmethod
    def _get_socket_mark(cls, sock, server):
        if server:
            port = sock.getpeername()[1]
        else:
            port = sock.getsockname()[1]
        return cls.MARK_FORMAT % (port,)