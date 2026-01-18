from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class Merge(TestBase):
    """Storage of versions that merge diverged parents"""

    def runTest(self):
        k = Weave()
        texts = [[b'header'], [b'header', b'', b'line from 1'], [b'header', b'', b'line from 2', b'more from 2'], [b'header', b'', b'line from 1', b'fixup line', b'line from 2']]
        k.add_lines(b'text0', [], texts[0])
        k.add_lines(b'text1', [b'text0'], texts[1])
        k.add_lines(b'text2', [b'text0'], texts[2])
        k.add_lines(b'merge', [b'text0', b'text1', b'text2'], texts[3])
        for i, t in enumerate(texts):
            self.assertEqual(k.get_lines(i), t)
        self.assertEqual(k.annotate(b'merge'), [(b'text0', b'header'), (b'text1', b''), (b'text1', b'line from 1'), (b'merge', b'fixup line'), (b'text2', b'line from 2')])
        self.assertEqual(set(k.get_ancestry([b'merge'])), {b'text0', b'text1', b'text2', b'merge'})
        self.log('k._weave=' + pformat(k._weave))
        self.check_read_write(k)