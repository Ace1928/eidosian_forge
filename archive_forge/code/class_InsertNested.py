from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class InsertNested(TestBase):
    """Insertion with nested instructions."""

    def runTest(self):
        k = Weave()
        k._parents = [frozenset(), frozenset([0]), frozenset([0]), frozenset([0, 1, 2])]
        k._weave = [(b'{', 0), b'foo {', (b'{', 1), b'  added in version 1', (b'{', 2), b'  added in v2', (b'}', 2), b'  also from v1', (b'}', 1), b'}', (b'}', 0)]
        k._sha1s = [sha_string(b'foo {}'), sha_string(b'foo {  added in version 1  also from v1}'), sha_string(b'foo {  added in v2}'), sha_string(b'foo {  added in version 1  added in v2  also from v1}')]
        self.assertEqual(k.get_lines(0), [b'foo {', b'}'])
        self.assertEqual(k.get_lines(1), [b'foo {', b'  added in version 1', b'  also from v1', b'}'])
        self.assertEqual(k.get_lines(2), [b'foo {', b'  added in v2', b'}'])
        self.assertEqual(k.get_lines(3), [b'foo {', b'  added in version 1', b'  added in v2', b'  also from v1', b'}'])