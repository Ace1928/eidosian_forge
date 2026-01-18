import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@classmethod
def _stop_server(cls):
    cls.io_loop.add_callback(cls.server.stop)
    cls.io_loop.add_callback(cls.io_loop.stop)
    cls.server_thread.join()