import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class ExecTestCase(unittest.TestCase):
    """A test case which runs external scripts for test fixtures."""

    def __init__(self, methodName='runTest'):
        """Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        """
        unittest.TestCase.__init__(self, methodName)
        testMethod = getattr(self, methodName)
        self.script = join_dir(sys.modules[self.__class__.__module__].__file__, testMethod.__doc__)

    def countTestCases(self):
        return 1

    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        self._run(result)

    def debug(self):
        """Run the test without collecting errors in a TestResult"""
        self._run(testresult.TestResult())

    def _run(self, result):
        protocol = TestProtocolServer(result)
        process = subprocess.Popen(self.script, shell=True, stdout=subprocess.PIPE)
        make_stream_binary(process.stdout)
        output = process.communicate()[0]
        protocol.readFrom(BytesIO(output))