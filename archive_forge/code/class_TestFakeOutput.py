import sys
import unittest
from unittest import mock
from bpython.curtsiesfrontend.coderunner import CodeRunner, FakeOutput
class TestFakeOutput(unittest.TestCase):

    def assert_unicode(self, s):
        self.assertIsInstance(s, str)

    def test_bytes(self):
        out = FakeOutput(mock.Mock(), self.assert_unicode, None)
        out.write('native string type')