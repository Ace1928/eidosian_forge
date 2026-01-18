import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
class HTTPDummyServerTestCase(object):
    """A simple HTTP server that runs when your test class runs

    Have your test class inherit from this one, and then a simple server
    will start when your tests run, and automatically shut down when they
    complete. For examples of what test requests you can send to the server,
    see the TestingApp in dummyserver/handlers.py.
    """
    scheme = 'http'
    host = 'localhost'
    host_alt = '127.0.0.1'
    certs = DEFAULT_CERTS

    @classmethod
    def _start_server(cls):
        cls.io_loop = ioloop.IOLoop.current()
        app = web.Application([('.*', TestingApp)])
        cls.server, cls.port = run_tornado_app(app, cls.io_loop, cls.certs, cls.scheme, cls.host)
        cls.server_thread = run_loop_in_thread(cls.io_loop)

    @classmethod
    def _stop_server(cls):
        cls.io_loop.add_callback(cls.server.stop)
        cls.io_loop.add_callback(cls.io_loop.stop)
        cls.server_thread.join()

    @classmethod
    def setup_class(cls):
        cls._start_server()

    @classmethod
    def teardown_class(cls):
        cls._stop_server()