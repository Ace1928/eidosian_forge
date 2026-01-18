import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
class CustomRNG(object):

    def __init__(self):
        self.counter = 0

    def __call__(self, size):
        self.counter += size
        return bchr(0) * size