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
class _ReadingSkipDetails(_ReadingDetails):
    """State for the subunit parser when reading skip details."""

    def _report_outcome(self):
        self.parser.client.addSkip(self.parser._current_test, details=self.details_parser.get_details('skip'))

    def _outcome_label(self):
        return 'skip'