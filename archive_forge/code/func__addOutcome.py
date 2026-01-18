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
def _addOutcome(self, outcome, test, error=None, details=None, error_permitted=True):
    """Report a failure in test test.

        Only one of error and details should be provided: conceptually there
        are two separate methods:
            addOutcome(self, test, error)
            addOutcome(self, test, details)

        :param outcome: A string describing the outcome - used as the
            event name in the subunit stream.
        :param error: Standard unittest positional argument form - an
            exc_info tuple.
        :param details: New Testing-in-python drafted API; a dict from string
            to subunit.Content objects.
        :param error_permitted: If True then one and only one of error or
            details must be supplied. If False then error must not be supplied
            and details is still optional.  """
    self._stream.write(_b('%s: ' % outcome) + self._test_id(test))
    if error_permitted:
        if error is None and details is None:
            raise ValueError
    elif error is not None:
        raise ValueError
    if error is not None:
        self._stream.write(self._start_simple)
        tb_content = TracebackContent(error, test)
        for bytes in tb_content.iter_bytes():
            self._stream.write(bytes)
    elif details is not None:
        self._write_details(details)
    else:
        self._stream.write(_b('\n'))
    if details is not None or error is not None:
        self._stream.write(self._end_simple)