from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class Test2a(tests.TestCaseWithMemoryTransport):

    def test_chk_bytes_uses_custom_btree_parser(self):
        mt = self.make_branch_and_memory_tree('test', format='2a')
        mt.lock_write()
        self.addCleanup(mt.unlock)
        mt.add([''], [b'root-id'])
        mt.commit('first')
        index = mt.branch.repository.chk_bytes._index._graph_index._indices[0]
        self.assertEqual(btree_index._gcchk_factory, index._leaf_factory)
        repo = mt.branch.repository.controldir.open_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        index = repo.chk_bytes._index._graph_index._indices[0]
        self.assertEqual(btree_index._gcchk_factory, index._leaf_factory)

    def test_fetch_combines_groups(self):
        builder = self.make_branch_builder('source', format='2a')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'1')
        builder.build_snapshot([b'1'], [('modify', ('file', b'content-2\n'))], revision_id=b'2')
        builder.finish_series()
        source = builder.get_branch()
        target = self.make_repository('target', format='2a')
        target.fetch(source.repository)
        target.lock_read()
        self.addCleanup(target.unlock)
        details = target.texts._index.get_build_details([(b'file-id', b'1'), (b'file-id', b'2')])
        file_1_details = details[b'file-id', b'1']
        file_2_details = details[b'file-id', b'2']
        self.assertEqual(file_1_details[0][:3], file_2_details[0][:3])

    def test_format_pack_compresses_True(self):
        repo = self.make_repository('repo', format='2a')
        self.assertTrue(repo._format.pack_compresses)

    def test_inventories_use_chk_map_with_parent_base_dict(self):
        tree = self.make_branch_and_memory_tree('repo', format='2a')
        tree.lock_write()
        tree.add([''], ids=[b'TREE_ROOT'])
        revid = tree.commit('foo')
        tree.unlock()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        inv = tree.branch.repository.get_inventory(revid)
        self.assertNotEqual(None, inv.parent_id_basename_to_file_id)
        inv.parent_id_basename_to_file_id._ensure_root()
        inv.id_to_entry._ensure_root()
        self.assertEqual(65536, inv.id_to_entry._root_node.maximum_size)
        self.assertEqual(65536, inv.parent_id_basename_to_file_id._root_node.maximum_size)

    def test_autopack_unchanged_chk_nodes(self):
        tree = self.make_branch_and_memory_tree('tree', format='2a')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ids=[b'TREE_ROOT'])
        for pos in range(20):
            tree.commit(str(pos))

    def test_pack_with_hint(self):
        tree = self.make_branch_and_memory_tree('tree', format='2a')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ids=[b'TREE_ROOT'])
        tree.commit('1')
        to_keep = tree.branch.repository._pack_collection.names()
        tree.commit('2')
        tree.commit('3')
        all = tree.branch.repository._pack_collection.names()
        combine = list(set(all) - set(to_keep))
        self.assertLength(3, all)
        self.assertLength(2, combine)
        tree.branch.repository.pack(hint=combine)
        final = tree.branch.repository._pack_collection.names()
        self.assertLength(2, final)
        self.assertFalse(combine[0] in final)
        self.assertFalse(combine[1] in final)
        self.assertSubset(to_keep, final)

    def test_stream_source_to_gc(self):
        source = self.make_repository('source', format='2a')
        target = self.make_repository('target', format='2a')
        stream = source._get_source(target._format)
        self.assertIsInstance(stream, groupcompress_repo.GroupCHKStreamSource)

    def test_stream_source_to_non_gc(self):
        source = self.make_repository('source', format='2a')
        target = self.make_repository('target', format='rich-root-pack')
        stream = source._get_source(target._format)
        self.assertIs(type(stream), vf_repository.StreamSource)

    def test_get_stream_for_missing_keys_includes_all_chk_refs(self):
        source_builder = self.make_branch_builder('source', format='2a')
        entries = [('add', ('', b'a-root-id', 'directory', None))]
        for i in 'abcdefghijklmnopqrstuvwxyz123456789':
            for j in 'abcdefghijklmnopqrstuvwxyz123456789':
                fname = i + j
                fid = fname.encode('utf-8') + b'-id'
                content = b'content for %s\n' % (fname.encode('utf-8'),)
                entries.append(('add', (fname, fid, 'file', content)))
        source_builder.start_series()
        source_builder.build_snapshot(None, entries, revision_id=b'rev-1')
        source_builder.build_snapshot([b'rev-1'], [('modify', ('aa', b'new content for aa-id\n')), ('modify', ('cc', b'new content for cc-id\n')), ('modify', ('zz', b'new content for zz-id\n'))], revision_id=b'rev-2')
        source_builder.finish_series()
        source_branch = source_builder.get_branch()
        source_branch.lock_read()
        self.addCleanup(source_branch.unlock)
        target = self.make_repository('target', format='2a')
        source = source_branch.repository._get_source(target._format)
        self.assertIsInstance(source, groupcompress_repo.GroupCHKStreamSource)
        search = vf_search.SearchResult({b'rev-2'}, {b'rev-1'}, 1, {b'rev-2'})
        simple_chk_records = set()
        for vf_name, substream in source.get_stream(search):
            if vf_name == 'chk_bytes':
                for record in substream:
                    simple_chk_records.add(record.key)
            else:
                for _ in substream:
                    continue
        self.assertEqual({(b'sha1:91481f539e802c76542ea5e4c83ad416bf219f73',), (b'sha1:4ff91971043668583985aec83f4f0ab10a907d3f',), (b'sha1:81e7324507c5ca132eedaf2d8414ee4bb2226187',), (b'sha1:b101b7da280596c71a4540e9a1eeba8045985ee0',)}, set(simple_chk_records))
        missing = [('inventories', b'rev-2')]
        full_chk_records = set()
        for vf_name, substream in source.get_stream_for_missing_keys(missing):
            if vf_name == 'inventories':
                for record in substream:
                    self.assertEqual((b'rev-2',), record.key)
            elif vf_name == 'chk_bytes':
                for record in substream:
                    full_chk_records.add(record.key)
            else:
                self.fail('Should not be getting a stream of {}'.format(vf_name))
        self.assertEqual(257, len(full_chk_records))
        self.assertSubset(simple_chk_records, full_chk_records)

    def test_inconsistency_fatal(self):
        repo = self.make_repository('repo', format='2a')
        self.assertTrue(repo.revisions._index._inconsistency_fatal)
        self.assertFalse(repo.texts._index._inconsistency_fatal)
        self.assertFalse(repo.inventories._index._inconsistency_fatal)
        self.assertFalse(repo.signatures._index._inconsistency_fatal)
        self.assertFalse(repo.chk_bytes._index._inconsistency_fatal)