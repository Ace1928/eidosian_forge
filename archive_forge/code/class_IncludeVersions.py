from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class IncludeVersions(TestBase):
    """Check texts that are stored across multiple revisions.

    Here we manually create a weave with particular encoding and make
    sure it unpacks properly.

    Text 0 includes nothing; text 1 includes text 0 and adds some
    lines.
    """

    def runTest(self):
        k = Weave()
        k._parents = [frozenset(), frozenset([0])]
        k._weave = [(b'{', 0), b'first line', (b'}', 0), (b'{', 1), b'second line', (b'}', 1)]
        k._sha1s = [sha_string(b'first line'), sha_string(b'first linesecond line')]
        self.assertEqual(k.get_lines(1), [b'first line', b'second line'])
        self.assertEqual(k.get_lines(0), [b'first line'])