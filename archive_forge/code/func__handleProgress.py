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
def _handleProgress(self, offset, line):
    """Process a progress directive."""
    line = line[offset:].strip()
    if line[0] in self._plusminus:
        whence = PROGRESS_CUR
        delta = int(line)
    elif line == self._push_sym:
        whence = PROGRESS_PUSH
        delta = None
    elif line == self._pop_sym:
        whence = PROGRESS_POP
        delta = None
    else:
        whence = PROGRESS_SET
        delta = int(line)
    self.client.progress(delta, whence)