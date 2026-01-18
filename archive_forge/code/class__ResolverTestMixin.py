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
class _ResolverTestMixin(object):
    resolver = None

    @gen_test
    def test_localhost(self: typing.Any):
        addrinfo = (yield self.resolver.resolve('localhost', 80, socket.AF_UNSPEC))
        self.assertTrue((socket.AF_INET, ('127.0.0.1', 80)) in addrinfo or (socket.AF_INET6, ('::1', 80)) in addrinfo, f'loopback address not found in {addrinfo}')