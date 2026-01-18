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
class IsValidIPTest(unittest.TestCase):

    def test_is_valid_ip(self):
        self.assertTrue(is_valid_ip('127.0.0.1'))
        self.assertTrue(is_valid_ip('4.4.4.4'))
        self.assertTrue(is_valid_ip('::1'))
        self.assertTrue(is_valid_ip('2620:0:1cfe:face:b00c::3'))
        self.assertTrue(not is_valid_ip('www.google.com'))
        self.assertTrue(not is_valid_ip('localhost'))
        self.assertTrue(not is_valid_ip('4.4.4.4<'))
        self.assertTrue(not is_valid_ip(' 127.0.0.1'))
        self.assertTrue(not is_valid_ip(''))
        self.assertTrue(not is_valid_ip(' '))
        self.assertTrue(not is_valid_ip('\n'))
        self.assertTrue(not is_valid_ip('\x00'))
        self.assertTrue(not is_valid_ip('a' * 100))