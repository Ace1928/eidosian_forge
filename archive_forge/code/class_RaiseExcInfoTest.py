from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
class RaiseExcInfoTest(unittest.TestCase):

    def test_two_arg_exception(self):

        class TwoArgException(Exception):

            def __init__(self, a, b):
                super().__init__()
                self.a, self.b = (a, b)
        try:
            raise TwoArgException(1, 2)
        except TwoArgException:
            exc_info = sys.exc_info()
        try:
            raise_exc_info(exc_info)
            self.fail("didn't get expected exception")
        except TwoArgException as e:
            self.assertIs(e, exc_info[1])