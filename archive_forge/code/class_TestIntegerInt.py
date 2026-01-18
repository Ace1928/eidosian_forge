import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
class TestIntegerInt(TestIntegerBase):

    def setUp(self):
        self.Integer = IntegerNative