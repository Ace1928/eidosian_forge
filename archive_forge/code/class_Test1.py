import io
import sys
import unittest
class Test1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        results.append('setup 1')

    @classmethod
    def tearDownClass(cls):
        results.append('teardown 1')

    def testOne(self):
        results.append('Test1.testOne')

    def testTwo(self):
        results.append('Test1.testTwo')