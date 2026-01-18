from __future__ import absolute_import
import unittest
import sys
import os
class NoExtensionTestSuite(unittest.TestSuite):

    def run(self, result):
        import simplejson
        simplejson._toggle_speedups(False)
        result = unittest.TestSuite.run(self, result)
        simplejson._toggle_speedups(True)
        return result