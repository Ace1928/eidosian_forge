from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class InvalidAdd(TestBase):
    """Try to use invalid version number during add."""

    def runTest(self):
        k = Weave()
        self.assertRaises(errors.RevisionNotPresent, k.add_lines, b'text0', [b'69'], [b'new text!'])