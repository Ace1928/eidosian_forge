import operator
import sys
import types
import unittest
import abc
import pytest
import six
class TestAssertNotRegex(unittest.TestCase):

    def test(self):
        with self.assertRaises(AssertionError):
            six.assertNotRegex(self, 'test', '^t')
        six.assertNotRegex(self, 'test', '^a')