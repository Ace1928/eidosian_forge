import io
import sys
import unittest
class BrokenTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raise TypeError('foo')

    def test_one(self):
        pass

    def test_two(self):
        pass