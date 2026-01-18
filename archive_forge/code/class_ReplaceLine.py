from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class ReplaceLine(TestBase):

    def runTest(self):
        k = Weave()
        text0 = [b'cheddar', b'stilton', b'gruyere']
        text1 = [b'cheddar', b'blue vein', b'neufchatel', b'chevre']
        k.add_lines(b'text0', [], text0)
        k.add_lines(b'text1', [b'text0'], text1)
        self.log('k._weave=' + pformat(k._weave))
        self.assertEqual(k.get_lines(0), text0)
        self.assertEqual(k.get_lines(1), text1)