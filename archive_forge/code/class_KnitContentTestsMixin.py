import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class KnitContentTestsMixin:

    def test_constructor(self):
        content = self._make_content([])

    def test_text(self):
        content = self._make_content([])
        self.assertEqual(content.text(), [])
        content = self._make_content([(b'origin1', b'text1'), (b'origin2', b'text2')])
        self.assertEqual(content.text(), [b'text1', b'text2'])

    def test_copy(self):
        content = self._make_content([(b'origin1', b'text1'), (b'origin2', b'text2')])
        copy = content.copy()
        self.assertIsInstance(copy, content.__class__)
        self.assertEqual(copy.annotate(), content.annotate())

    def assertDerivedBlocksEqual(self, source, target, noeol=False):
        """Assert that the derived matching blocks match real output"""
        source_lines = source.splitlines(True)
        target_lines = target.splitlines(True)

        def nl(line):
            if noeol and (not line.endswith('\n')):
                return line + '\n'
            else:
                return line
        source_content = self._make_content([(None, nl(l)) for l in source_lines])
        target_content = self._make_content([(None, nl(l)) for l in target_lines])
        line_delta = source_content.line_delta(target_content)
        delta_blocks = list(KnitContent.get_line_delta_blocks(line_delta, source_lines, target_lines))
        matcher = PatienceSequenceMatcher(None, source_lines, target_lines)
        matcher_blocks = list(matcher.get_matching_blocks())
        self.assertEqual(matcher_blocks, delta_blocks)

    def test_get_line_delta_blocks(self):
        self.assertDerivedBlocksEqual('a\nb\nc\n', 'q\nc\n')
        self.assertDerivedBlocksEqual(TEXT_1, TEXT_1)
        self.assertDerivedBlocksEqual(TEXT_1, TEXT_1A)
        self.assertDerivedBlocksEqual(TEXT_1, TEXT_1B)
        self.assertDerivedBlocksEqual(TEXT_1B, TEXT_1A)
        self.assertDerivedBlocksEqual(TEXT_1A, TEXT_1B)
        self.assertDerivedBlocksEqual(TEXT_1A, '')
        self.assertDerivedBlocksEqual('', TEXT_1A)
        self.assertDerivedBlocksEqual('', '')
        self.assertDerivedBlocksEqual('a\nb\nc', 'a\nb\nc\nd')

    def test_get_line_delta_blocks_noeol(self):
        """Handle historical knit deltas safely

        Some existing knit deltas don't consider the last line to differ
        when the only difference whether it has a final newline.

        New knit deltas appear to always consider the last line to differ
        in this case.
        """
        self.assertDerivedBlocksEqual('a\nb\nc', 'a\nb\nc\nd\n', noeol=True)
        self.assertDerivedBlocksEqual('a\nb\nc\nd\n', 'a\nb\nc', noeol=True)
        self.assertDerivedBlocksEqual('a\nb\nc\n', 'a\nb\nc', noeol=True)
        self.assertDerivedBlocksEqual('a\nb\nc', 'a\nb\nc\n', noeol=True)