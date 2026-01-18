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
@unittest.skipIf(twisted is None, 'twisted module not present')
@unittest.skipIf(getattr(twisted, '__version__', '0.0') < '12.1', 'old version of twisted')
@unittest.skipIf(sys.platform == 'win32', 'twisted resolver hangs on windows')
class TwistedResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        super().setUp()
        self.resolver = TwistedResolver()