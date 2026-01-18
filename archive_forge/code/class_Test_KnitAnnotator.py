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
class Test_KnitAnnotator(TestCaseWithMemoryTransport):

    def make_annotator(self):
        factory = knit.make_pack_factory(True, True, 1)
        vf = factory(self.get_transport())
        return knit._KnitAnnotator(vf)

    def test__expand_fulltext(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        ann._num_compression_children[rev_key] = 1
        res = ann._expand_record(rev_key, ((b'parent-id',),), None, [b'line1\n', b'line2\n'], ('fulltext', True))
        self.assertEqual([b'line1\n', b'line2'], res)
        content_obj = ann._content_objects[rev_key]
        self.assertEqual([b'line1\n', b'line2\n'], content_obj._lines)
        self.assertEqual(res, content_obj.text())
        self.assertEqual(res, ann._text_cache[rev_key])

    def test__expand_delta_comp_parent_not_available(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        parent_key = (b'parent-id',)
        record = [b'0,1,1\n', b'new-line\n']
        details = ('line-delta', False)
        res = ann._expand_record(rev_key, (parent_key,), parent_key, record, details)
        self.assertEqual(None, res)
        self.assertTrue(parent_key in ann._pending_deltas)
        pending = ann._pending_deltas[parent_key]
        self.assertEqual(1, len(pending))
        self.assertEqual((rev_key, (parent_key,), record, details), pending[0])

    def test__expand_record_tracks_num_children(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        rev2_key = (b'rev2-id',)
        parent_key = (b'parent-id',)
        record = [b'0,1,1\n', b'new-line\n']
        details = ('line-delta', False)
        ann._num_compression_children[parent_key] = 2
        ann._expand_record(parent_key, (), None, [b'line1\n', b'line2\n'], ('fulltext', False))
        res = ann._expand_record(rev_key, (parent_key,), parent_key, record, details)
        self.assertEqual({parent_key: 1}, ann._num_compression_children)
        res = ann._expand_record(rev2_key, (parent_key,), parent_key, record, details)
        self.assertFalse(parent_key in ann._content_objects)
        self.assertEqual({}, ann._num_compression_children)
        self.assertEqual({}, ann._content_objects)

    def test__expand_delta_records_blocks(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        parent_key = (b'parent-id',)
        record = [b'0,1,1\n', b'new-line\n']
        details = ('line-delta', True)
        ann._num_compression_children[parent_key] = 2
        ann._expand_record(parent_key, (), None, [b'line1\n', b'line2\n', b'line3\n'], ('fulltext', False))
        ann._expand_record(rev_key, (parent_key,), parent_key, record, details)
        self.assertEqual({(rev_key, parent_key): [(1, 1, 1), (3, 3, 0)]}, ann._matching_blocks)
        rev2_key = (b'rev2-id',)
        record = [b'0,1,1\n', b'new-line\n']
        details = ('line-delta', False)
        ann._expand_record(rev2_key, (parent_key,), parent_key, record, details)
        self.assertEqual([(1, 1, 2), (3, 3, 0)], ann._matching_blocks[rev2_key, parent_key])

    def test__get_parent_ann_uses_matching_blocks(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        parent_key = (b'parent-id',)
        parent_ann = [(parent_key,)] * 3
        block_key = (rev_key, parent_key)
        ann._annotations_cache[parent_key] = parent_ann
        ann._matching_blocks[block_key] = [(0, 1, 1), (3, 3, 0)]
        par_ann, blocks = ann._get_parent_annotations_and_matches(rev_key, [b'1\n', b'2\n', b'3\n'], parent_key)
        self.assertEqual(parent_ann, par_ann)
        self.assertEqual([(0, 1, 1), (3, 3, 0)], blocks)
        self.assertEqual({}, ann._matching_blocks)

    def test__process_pending(self):
        ann = self.make_annotator()
        rev_key = (b'rev-id',)
        p1_key = (b'p1-id',)
        p2_key = (b'p2-id',)
        record = [b'0,1,1\n', b'new-line\n']
        details = ('line-delta', False)
        p1_record = [b'line1\n', b'line2\n']
        ann._num_compression_children[p1_key] = 1
        res = ann._expand_record(rev_key, (p1_key, p2_key), p1_key, record, details)
        self.assertEqual(None, res)
        self.assertEqual({}, ann._pending_annotation)
        res = ann._expand_record(p1_key, (), None, p1_record, ('fulltext', False))
        self.assertEqual(p1_record, res)
        ann._annotations_cache[p1_key] = [(p1_key,)] * 2
        res = ann._process_pending(p1_key)
        self.assertEqual([], res)
        self.assertFalse(p1_key in ann._pending_deltas)
        self.assertTrue(p2_key in ann._pending_annotation)
        self.assertEqual({p2_key: [(rev_key, (p1_key, p2_key))]}, ann._pending_annotation)
        res = ann._expand_record(p2_key, (), None, [], ('fulltext', False))
        ann._annotations_cache[p2_key] = []
        res = ann._process_pending(p2_key)
        self.assertEqual([rev_key], res)
        self.assertEqual({}, ann._pending_annotation)
        self.assertEqual({}, ann._pending_deltas)

    def test_record_delta_removes_basis(self):
        ann = self.make_annotator()
        ann._expand_record((b'parent-id',), (), None, [b'line1\n', b'line2\n'], ('fulltext', False))
        ann._num_compression_children[b'parent-id'] = 2

    def test_annotate_special_text(self):
        ann = self.make_annotator()
        vf = ann._vf
        rev1_key = (b'rev-1',)
        rev2_key = (b'rev-2',)
        rev3_key = (b'rev-3',)
        spec_key = (b'special:',)
        vf.add_lines(rev1_key, [], [b'initial content\n'])
        vf.add_lines(rev2_key, [rev1_key], [b'initial content\n', b'common content\n', b'content in 2\n'])
        vf.add_lines(rev3_key, [rev1_key], [b'initial content\n', b'common content\n', b'content in 3\n'])
        spec_text = b'initial content\ncommon content\ncontent in 2\ncontent in 3\n'
        ann.add_special_text(spec_key, [rev2_key, rev3_key], spec_text)
        anns, lines = ann.annotate(spec_key)
        self.assertEqual([(rev1_key,), (rev2_key, rev3_key), (rev2_key,), (rev3_key,)], anns)
        self.assertEqualDiff(spec_text, b''.join(lines))