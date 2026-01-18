import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class NoInputEncodingTestCase(unittest.TestCase):

    def setUp(self):
        self.old_stdin = sys.stdin

        class X:
            pass
        fake_stdin = X()
        sys.stdin = fake_stdin

    def test(self):
        enc = isp.get_input_encoding()
        self.assertEqual(enc, 'ascii')

    def tearDown(self):
        sys.stdin = self.old_stdin