import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
class FakeTP(unittest.TestProgram):

    def parseArgs(self, *args, **kw):
        pass

    def runTests(self, *args, **kw):
        pass