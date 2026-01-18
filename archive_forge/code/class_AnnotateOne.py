from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class AnnotateOne(TestBase):

    def runTest(self):
        k = Weave()
        k.add_lines(b'text0', [], TEXT_0)
        self.assertEqual(k.annotate(b'text0'), [(b'text0', TEXT_0[0])])