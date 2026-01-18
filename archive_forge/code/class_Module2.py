import io
import sys
import unittest
class Module2(object):

    @staticmethod
    def setUpModule():
        results.append('Module2.setUpModule')

    @staticmethod
    def tearDownModule():
        results.append('Module2.tearDownModule')