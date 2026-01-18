import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestGroupCompressVersionedFiles(TestCaseWithGroupCompressVersionedFiles):

    def make_g_index(self, name, ref_lists=0, nodes=[]):
        builder = btree_index.BTreeBuilder(ref_lists)
        for node, references, value in nodes:
            builder.add_node(node, references, value)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file(name, stream)
        return btree_index.BTreeGraphIndex(trans, name, size)

    def make_g_index_missing_parent(self):
        graph_index = self.make_g_index('missing_parent', 1, [((b'parent',), b'2 78 2 10', ([],)), ((b'tip',), b'2 78 2 10', ([(b'parent',), (b'missing-parent',)],))])
        return graph_index

    def test_get_record_stream_as_requested(self):
        vf = self.make_test_vf(False, dir='source')
        vf.add_lines((b'a',), (), [b'lines\n'])
        vf.add_lines((b'b',), (), [b'lines\n'])
        vf.add_lines((b'c',), (), [b'lines\n'])
        vf.add_lines((b'd',), (), [b'lines\n'])
        vf.writer.end()
        keys = [record.key for record in vf.get_record_stream([(b'a',), (b'b',), (b'c',), (b'd',)], 'as-requested', False)]
        self.assertEqual([(b'a',), (b'b',), (b'c',), (b'd',)], keys)
        keys = [record.key for record in vf.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False)]
        self.assertEqual([(b'b',), (b'a',), (b'd',), (b'c',)], keys)
        vf2 = self.make_test_vf(False, dir='target')
        vf2.insert_record_stream(vf.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False))
        vf2.writer.end()
        keys = [record.key for record in vf2.get_record_stream([(b'a',), (b'b',), (b'c',), (b'd',)], 'as-requested', False)]
        self.assertEqual([(b'a',), (b'b',), (b'c',), (b'd',)], keys)
        keys = [record.key for record in vf2.get_record_stream([(b'b',), (b'a',), (b'd',), (b'c',)], 'as-requested', False)]
        self.assertEqual([(b'b',), (b'a',), (b'd',), (b'c',)], keys)

    def test_get_record_stream_max_bytes_to_index_default(self):
        vf = self.make_test_vf(True, dir='source')
        vf.add_lines((b'a',), (), [b'lines\n'])
        vf.writer.end()
        record = next(vf.get_record_stream([(b'a',)], 'unordered', True))
        self.assertEqual(vf._DEFAULT_COMPRESSOR_SETTINGS, record._manager._get_compressor_settings())

    def test_get_record_stream_accesses_compressor_settings(self):
        vf = self.make_test_vf(True, dir='source')
        vf.add_lines((b'a',), (), [b'lines\n'])
        vf.writer.end()
        vf._max_bytes_to_index = 1234
        record = next(vf.get_record_stream([(b'a',)], 'unordered', True))
        self.assertEqual(dict(max_bytes_to_index=1234), record._manager._get_compressor_settings())

    @staticmethod
    def grouped_stream(revision_ids, first_parents=()):
        parents = first_parents
        for revision_id in revision_ids:
            key = (revision_id,)
            record = versionedfile.FulltextContentFactory(key, parents, None, b'some content that is\nidentical except for\nrevision_id:%s\n' % (revision_id,))
            yield record
            parents = (key,)

    def test_insert_record_stream_reuses_blocks(self):
        vf = self.make_test_vf(True, dir='source')
        vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
        vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
        block_bytes = {}
        stream = vf.get_record_stream([(r.encode(),) for r in 'abcdefgh'], 'unordered', False)
        num_records = 0
        for record in stream:
            if record.key in [(b'a',), (b'e',)]:
                self.assertEqual('groupcompress-block', record.storage_kind)
            else:
                self.assertEqual('groupcompress-block-ref', record.storage_kind)
            block_bytes[record.key] = record._manager._block._z_content
            num_records += 1
        self.assertEqual(8, num_records)
        for r in 'abcd':
            key = (r.encode(),)
            self.assertIs(block_bytes[key], block_bytes[b'a',])
            self.assertNotEqual(block_bytes[key], block_bytes[b'e',])
        for r in 'efgh':
            key = (r.encode(),)
            self.assertIs(block_bytes[key], block_bytes[b'e',])
            self.assertNotEqual(block_bytes[key], block_bytes[b'a',])
        vf2 = self.make_test_vf(True, dir='target')
        keys = [(r.encode(),) for r in 'abcdefgh']

        def small_size_stream():
            for record in vf.get_record_stream(keys, 'groupcompress', False):
                record._manager._full_enough_block_size = record._manager._block._content_length
                yield record
        vf2.insert_record_stream(small_size_stream())
        stream = vf2.get_record_stream(keys, 'groupcompress', False)
        vf2.writer.end()
        num_records = 0
        for record in stream:
            num_records += 1
            self.assertEqual(block_bytes[record.key], record._manager._block._z_content)
        self.assertEqual(8, num_records)

    def test_insert_record_stream_packs_on_the_fly(self):
        vf = self.make_test_vf(True, dir='source')
        vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
        vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
        vf2 = self.make_test_vf(True, dir='target')
        keys = [(r.encode(),) for r in 'abcdefgh']
        vf2.insert_record_stream(vf.get_record_stream(keys, 'groupcompress', False))
        stream = vf2.get_record_stream(keys, 'groupcompress', False)
        vf2.writer.end()
        num_records = 0
        block = None
        for record in stream:
            num_records += 1
            if block is None:
                block = record._manager._block
            else:
                self.assertIs(block, record._manager._block)
        self.assertEqual(8, num_records)

    def test__insert_record_stream_no_reuse_block(self):
        vf = self.make_test_vf(True, dir='source')
        vf.insert_record_stream(self.grouped_stream([b'a', b'b', b'c', b'd']))
        vf.insert_record_stream(self.grouped_stream([b'e', b'f', b'g', b'h'], first_parents=((b'd',),)))
        vf.writer.end()
        keys = [(r.encode(),) for r in 'abcdefgh']
        self.assertEqual(8, len(list(vf.get_record_stream(keys, 'unordered', False))))
        vf2 = self.make_test_vf(True, dir='target')
        list(vf2._insert_record_stream(vf.get_record_stream(keys, 'groupcompress', False), reuse_blocks=False))
        vf2.writer.end()
        stream = vf2.get_record_stream(keys, 'groupcompress', False)
        block = None
        for record in stream:
            if block is None:
                block = record._manager._block
            else:
                self.assertIs(block, record._manager._block)

    def test_add_missing_noncompression_parent_unvalidated_index(self):
        unvalidated = self.make_g_index_missing_parent()
        combined = _mod_index.CombinedGraphIndex([unvalidated])
        index = groupcompress._GCGraphIndex(combined, is_locked=lambda: True, parents=True, track_external_parent_refs=True)
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset([(b'missing-parent',)]), index.get_missing_parents())

    def test_track_external_parent_refs(self):
        g_index = self.make_g_index('empty', 1, [])
        mod_index = btree_index.BTreeBuilder(1, 1)
        combined = _mod_index.CombinedGraphIndex([g_index, mod_index])
        index = groupcompress._GCGraphIndex(combined, is_locked=lambda: True, parents=True, add_callback=mod_index.add_nodes, track_external_parent_refs=True)
        index.add_records([((b'new-key',), b'2 10 2 10', [((b'parent-1',), (b'parent-2',))])])
        self.assertEqual(frozenset([(b'parent-1',), (b'parent-2',)]), index.get_missing_parents())

    def make_source_with_b(self, a_parent, path):
        source = self.make_test_vf(True, dir=path)
        source.add_lines((b'a',), (), [b'lines\n'])
        if a_parent:
            b_parents = ((b'a',),)
        else:
            b_parents = ()
        source.add_lines((b'b',), b_parents, [b'lines\n'])
        return source

    def do_inconsistent_inserts(self, inconsistency_fatal):
        target = self.make_test_vf(True, dir='target', inconsistency_fatal=inconsistency_fatal)
        for x in range(2):
            source = self.make_source_with_b(x == 1, 'source%s' % x)
            target.insert_record_stream(source.get_record_stream([(b'b',)], 'unordered', False))

    def test_inconsistent_redundant_inserts_warn(self):
        """Should not insert a record that is already present."""
        warnings = []

        def warning(template, args):
            warnings.append(template % args)
        _trace_warning = trace.warning
        trace.warning = warning
        try:
            self.do_inconsistent_inserts(inconsistency_fatal=False)
        finally:
            trace.warning = _trace_warning
        self.assertContainsRe('\n'.join(warnings), "^inconsistent details in skipped record: \\(b?'b',\\) \\(b?'42 32 0 8', \\(\\(\\),\\)\\) \\(b?'74 32 0 8', \\(\\(\\(b?'a',\\),\\),\\)\\)$")

    def test_inconsistent_redundant_inserts_raises(self):
        e = self.assertRaises(knit.KnitCorrupt, self.do_inconsistent_inserts, inconsistency_fatal=True)
        self.assertContainsRe(str(e), "Knit.* corrupt: inconsistent details in add_records: \\(b?'b',\\) \\(b?'42 32 0 8', \\(\\(\\),\\)\\) \\(b?'74 32 0 8', \\(\\(\\(b?'a',\\),\\),\\)\\)")

    def test_clear_cache(self):
        vf = self.make_source_with_b(True, 'source')
        vf.writer.end()
        for record in vf.get_record_stream([(b'a',), (b'b',)], 'unordered', True):
            pass
        self.assertTrue(len(vf._group_cache) > 0)
        vf.clear_cache()
        self.assertEqual(0, len(vf._group_cache))