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
class ThreadedResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        super().setUp()
        self.resolver = ThreadedResolver()

    def tearDown(self):
        self.resolver.close()
        super().tearDown()