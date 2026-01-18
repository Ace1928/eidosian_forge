from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class WeaveContains(TestBase):
    """Weave __contains__ operator"""

    def runTest(self):
        k = Weave(get_scope=lambda: None)
        self.assertFalse(b'foo' in k)
        k.add_lines(b'foo', [], TEXT_1)
        self.assertTrue(b'foo' in k)