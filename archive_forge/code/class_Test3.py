import io
import sys
import unittest
class Test3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        results.append('setup 3')

    @classmethod
    def tearDownClass(cls):
        results.append('teardown 3')

    def testOne(self):
        results.append('Test3.testOne')

    def testTwo(self):
        results.append('Test3.testTwo')