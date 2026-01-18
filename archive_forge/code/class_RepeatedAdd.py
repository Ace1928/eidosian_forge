from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class RepeatedAdd(TestBase):
    """Add the same version twice; harmless."""

    def test_duplicate_add(self):
        k = Weave()
        idx = k.add_lines(b'text0', [], TEXT_0)
        idx2 = k.add_lines(b'text0', [], TEXT_0)
        self.assertEqual(idx, idx2)