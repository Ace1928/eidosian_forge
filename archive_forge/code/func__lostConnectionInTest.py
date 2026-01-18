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
def _lostConnectionInTest(self, state_string):
    error_string = _u("lost connection during %stest '%s'") % (state_string, self.current_test_description)
    self.client.addError(self._current_test, RemoteError(error_string))
    self.client.stopTest(self._current_test)