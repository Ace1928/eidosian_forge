from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
class FixtureSuite(unittest.TestSuite):

    def __init__(self, fixture, tests):
        super().__init__(tests)
        self._fixture = fixture

    def run(self, result):
        self._fixture.setUp()
        try:
            super().run(result)
        finally:
            self._fixture.cleanUp()

    def sort_tests(self):
        self._tests = sorted_tests(self, True)