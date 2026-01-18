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
class _InTest(_ParserState):
    """State for the subunit parser after reading a test: directive."""

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

    def _error(self):
        self.parser.client.addError(self.parser._current_test, details={})

    def addError(self, offset, line):
        """An 'error:' directive has been read."""
        self._outcome(offset, line, self._error, self.parser._reading_error_details)

    def _xfail(self):
        self.parser.client.addExpectedFailure(self.parser._current_test, details={})

    def addExpectedFail(self, offset, line):
        """An 'xfail:' directive has been read."""
        self._outcome(offset, line, self._xfail, self.parser._reading_xfail_details)

    def _uxsuccess(self):
        self.parser.client.addUnexpectedSuccess(self.parser._current_test)

    def addUnexpectedSuccess(self, offset, line):
        """A 'uxsuccess:' directive has been read."""
        self._outcome(offset, line, self._uxsuccess, self.parser._reading_uxsuccess_details)

    def _failure(self):
        self.parser.client.addFailure(self.parser._current_test, details={})

    def addFailure(self, offset, line):
        """A 'failure:' directive has been read."""
        self._outcome(offset, line, self._failure, self.parser._reading_failure_details)

    def _skip(self):
        self.parser.client.addSkip(self.parser._current_test, details={})

    def addSkip(self, offset, line):
        """A 'skip:' directive has been read."""
        self._outcome(offset, line, self._skip, self.parser._reading_skip_details)

    def _succeed(self):
        self.parser.client.addSuccess(self.parser._current_test, details={})

    def addSuccess(self, offset, line):
        """A 'success:' directive has been read."""
        self._outcome(offset, line, self._succeed, self.parser._reading_success_details)

    def lostConnection(self):
        """Connection lost."""
        self.parser._lostConnectionInTest(_u(''))