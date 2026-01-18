import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
class LoggingCase(unittest.TestCase):

    def run(self, result):
        events.append('run %s' % self._testMethodName)

    def test1(self):
        pass

    def test2(self):
        pass