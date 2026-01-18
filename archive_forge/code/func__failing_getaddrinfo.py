import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import (
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
def _failing_getaddrinfo(*args):
    """Dummy implementation of getaddrinfo for use in mocks"""
    raise socket.gaierror(errno.EIO, 'mock: lookup failed')