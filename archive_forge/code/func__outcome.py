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
def _outcome(self, offset, line, no_details, details_state):
    """An outcome directive has been read.

        :param no_details: Callable to call when no details are presented.
        :param details_state: The state to switch to for details
            processing of this outcome.
        """
    test_name = line[offset:-1].decode('utf8')
    if self.parser.current_test_description == test_name:
        self.parser._state = self.parser._outside_test
        self.parser.current_test_description = None
        no_details()
        self.parser.client.stopTest(self.parser._current_test)
        self.parser._current_test = None
        self.parser.subunitLineReceived(line)
    elif self.parser.current_test_description + self._start_simple == test_name:
        self.parser._state = details_state
        details_state.set_simple()
        self.parser.subunitLineReceived(line)
    elif self.parser.current_test_description + self._start_multipart == test_name:
        self.parser._state = details_state
        details_state.set_multipart()
        self.parser.subunitLineReceived(line)
    else:
        self.parser.stdOutLineReceived(line)