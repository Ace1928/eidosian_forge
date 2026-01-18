from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class InsertLines(TestBase):
    """Store a revision that adds one line to the original.

    Look at the annotations to make sure that the first line is matched
    and not stored repeatedly."""

    def runTest(self):
        k = Weave()
        k.add_lines(b'text0', [], [b'line 1'])
        k.add_lines(b'text1', [b'text0'], [b'line 1', b'line 2'])
        self.assertEqual(k.annotate(b'text0'), [(b'text0', b'line 1')])
        self.assertEqual(k.get_lines(1), [b'line 1', b'line 2'])
        self.assertEqual(k.annotate(b'text1'), [(b'text0', b'line 1'), (b'text1', b'line 2')])
        k.add_lines(b'text2', [b'text0'], [b'line 1', b'diverged line'])
        self.assertEqual(k.annotate(b'text2'), [(b'text0', b'line 1'), (b'text2', b'diverged line')])
        text3 = [b'line 1', b'middle line', b'line 2']
        k.add_lines(b'text3', [b'text0', b'text1'], text3)
        self.log('k._weave=' + pformat(k._weave))
        self.assertEqual(k.annotate(b'text3'), [(b'text0', b'line 1'), (b'text3', b'middle line'), (b'text1', b'line 2')])
        k.add_lines(b'text4', [b'text0', b'text1', b'text3'], [b'line 1', b'aaa', b'middle line', b'bbb', b'line 2', b'ccc'])
        self.assertEqual(k.annotate(b'text4'), [(b'text0', b'line 1'), (b'text4', b'aaa'), (b'text3', b'middle line'), (b'text4', b'bbb'), (b'text1', b'line 2'), (b'text4', b'ccc')])