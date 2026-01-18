from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class BadInsert(TestBase):
    """Test that we trap an insert which should not occur."""

    def runTest(self):
        k = Weave()
        k._parents = [frozenset(), frozenset([0]), frozenset([0]), frozenset([0, 1, 2])]
        k._weave = [(b'{', 0), b'foo {', (b'{', 1), b'  added in version 1', (b'{', 1), b'  more in 1', (b'}', 1), (b'}', 1), (b'}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get, 0)
        self.assertRaises(WeaveFormatError, k.get, 1)