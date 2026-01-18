import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
class TestAssertPrints(unittest.TestCase):

    def test_passing(self):
        with tt.AssertPrints('abc'):
            print('abcd')
            print('def')
            print(b'ghi')

    def test_failing(self):

        def func():
            with tt.AssertPrints('abc'):
                print('acd')
                print('def')
                print(b'ghi')
        self.assertRaises(AssertionError, func)