import operator
import sys
import types
import unittest
import abc
import pytest
import six
class TestAssertCountEqual(unittest.TestCase):

    def test(self):
        with self.assertRaises(AssertionError):
            six.assertCountEqual(self, (1, 2), [3, 4, 5])
        six.assertCountEqual(self, (1, 2), [2, 1])