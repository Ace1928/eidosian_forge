from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class Conflicts(TestBase):
    """Test detection of conflicting regions during a merge.

    A base version is inserted, then two descendents try to
    insert different lines in the same place.  These should be
    reported as a possible conflict and forwarded to the user."""

    def runTest(self):
        return
        k = Weave()
        k.add_lines([], [b'aaa', b'bbb'])
        k.add_lines([0], [b'aaa', b'111', b'bbb'])
        k.add_lines([1], [b'aaa', b'222', b'bbb'])
        k.merge([1, 2])
        self.assertEqual([[[b'aaa']], [[b'111'], [b'222']], [[b'bbb']]])