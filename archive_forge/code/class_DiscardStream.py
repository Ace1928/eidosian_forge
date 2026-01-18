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
class DiscardStream(object):
    """A filelike object which discards what is written to it."""

    def fileno(self):
        raise _UnsupportedOperation()

    def write(self, bytes):
        pass

    def read(self, len=0):
        return _b('')