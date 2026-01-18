from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class NonConflict(TestBase):
    """Two descendants insert compatible changes.

    No conflict should be reported."""

    def runTest(self):
        return
        k = Weave()
        k.add_lines([], [b'aaa', b'bbb'])
        k.add_lines([0], [b'111', b'aaa', b'ccc', b'bbb'])
        k.add_lines([1], [b'aaa', b'ccc', b'bbb', b'222'])