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
class IsolatedTestCase(unittest.TestCase):
    """A TestCase which executes in a forked process.

    Each test gets its own process, which has a performance overhead but will
    provide excellent isolation from global state (such as django configs,
    zope utilities and so on).
    """

    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        run_isolated(unittest.TestCase, self, result)