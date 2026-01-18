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
@skipIfNoNetwork
@unittest.skipIf(pycares is None, 'pycares module not present')
@unittest.skipIf(sys.platform == 'win32', "pycares doesn't return loopback on windows")
@unittest.skipIf(sys.platform == 'darwin', "pycares doesn't return 127.0.0.1 on darwin")
class CaresResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        super().setUp()
        self.resolver = CaresResolver()