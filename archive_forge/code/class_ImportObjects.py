import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
class ImportObjects(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self._mapping = BzrGitMappingv1()
        factory = knit.make_file_factory(True, versionedfile.PrefixMapper())
        self._texts = factory(self.get_transport('texts'))

    def test_import_blob_missing_in_one_parent(self):
        builder = self.make_branch_builder('br')
        builder.start_series()
        rev_root = builder.build_snapshot(None, [('add', ('', b'rootid', 'directory', ''))])
        rev1 = builder.build_snapshot([rev_root], [('add', ('bla', self._mapping.generate_file_id('bla'), 'file', b'content'))])
        rev2 = builder.build_snapshot([rev_root], [])
        builder.finish_series()
        branch = builder.get_branch()
        blob = Blob.from_string(b'bar')
        objs = {'blobname': blob}
        ret = import_git_blob(self._texts, self._mapping, b'bla', b'bla', (None, 'blobname'), branch.repository.revision_tree(rev1), b'rootid', b'somerevid', [branch.repository.revision_tree(r) for r in [rev1, rev2]], objs.__getitem__, (None, DEFAULT_FILE_MODE), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual({(b'git:bla', b'somerevid')}, self._texts.keys())

    def test_import_blob_simple(self):
        blob = Blob.from_string(b'bar')
        objs = {'blobname': blob}
        ret = import_git_blob(self._texts, self._mapping, b'bla', b'bla', (None, 'blobname'), None, None, b'somerevid', [], objs.__getitem__, (None, DEFAULT_FILE_MODE), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual({(b'git:bla', b'somerevid')}, self._texts.keys())
        self.assertEqual(next(self._texts.get_record_stream([(b'git:bla', b'somerevid')], 'unordered', True)).get_bytes_as('fulltext'), b'bar')
        self.assertEqual(1, len(ret))
        self.assertEqual(None, ret[0][0])
        self.assertEqual('bla', ret[0][1])
        ie = ret[0][3]
        self.assertEqual(False, ie.executable)
        self.assertEqual('file', ie.kind)
        self.assertEqual(b'somerevid', ie.revision)
        self.assertEqual(osutils.sha_strings([b'bar']), ie.text_sha1)

    def test_import_tree_empty_root(self):
        tree = Tree()
        ret, child_modes = import_git_tree(self._texts, self._mapping, b'', b'', (None, tree.id), None, None, b'somerevid', [], {tree.id: tree}.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {})
        self.assertEqual({(b'TREE_ROOT', b'somerevid')}, self._texts.keys())
        self.assertEqual(1, len(ret))
        self.assertEqual(None, ret[0][0])
        self.assertEqual('', ret[0][1])
        ie = ret[0][3]
        self.assertEqual(False, ie.executable)
        self.assertEqual('directory', ie.kind)
        self.assertEqual({}, ie.children)
        self.assertEqual(b'somerevid', ie.revision)
        self.assertEqual(None, ie.text_sha1)

    def test_import_tree_empty(self):
        tree = Tree()
        ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], {tree.id: tree}.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {})
        self.assertEqual({(b'git:bla', b'somerevid')}, self._texts.keys())
        self.assertEqual(1, len(ret))
        self.assertEqual(None, ret[0][0])
        self.assertEqual('bla', ret[0][1])
        ie = ret[0][3]
        self.assertEqual('directory', ie.kind)
        self.assertEqual(False, ie.executable)
        self.assertEqual({}, ie.children)
        self.assertEqual(b'somerevid', ie.revision)
        self.assertEqual(None, ie.text_sha1)

    def test_import_tree_with_file(self):
        blob = Blob.from_string(b'bar1')
        tree = Tree()
        tree.add(b'foo', stat.S_IFREG | 420, blob.id)
        objects = {blob.id: blob, tree.id: tree}
        ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], objects.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {})
        self.assertEqual(2, len(ret))
        self.assertEqual(None, ret[0][0])
        self.assertEqual('bla', ret[0][1])
        self.assertEqual(None, ret[1][0])
        self.assertEqual('bla/foo', ret[1][1])
        ie = ret[0][3]
        self.assertEqual('directory', ie.kind)
        ie = ret[1][3]
        self.assertEqual('file', ie.kind)
        self.assertEqual(b'git:bla/foo', ie.file_id)
        self.assertEqual(b'somerevid', ie.revision)
        self.assertEqual(osutils.sha_strings([b'bar1']), ie.text_sha1)
        self.assertEqual(False, ie.executable)

    def test_import_tree_with_unusual_mode_file(self):
        blob = Blob.from_string(b'bar1')
        tree = Tree()
        tree.add(b'foo', stat.S_IFREG | 436, blob.id)
        objects = {blob.id: blob, tree.id: tree}
        ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], objects.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {b'bla/foo': stat.S_IFREG | 436})

    def test_import_tree_with_file_exe(self):
        blob = Blob.from_string(b'bar')
        tree = Tree()
        tree.add(b'foo', 33261, blob.id)
        objects = {blob.id: blob, tree.id: tree}
        ret, child_modes = import_git_tree(self._texts, self._mapping, b'', b'', (None, tree.id), None, None, b'somerevid', [], objects.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {})
        self.assertEqual(2, len(ret))
        self.assertEqual(None, ret[0][0])
        self.assertEqual('', ret[0][1])
        self.assertEqual(None, ret[1][0])
        self.assertEqual('foo', ret[1][1])
        ie = ret[0][3]
        self.assertEqual('directory', ie.kind)
        ie = ret[1][3]
        self.assertEqual('file', ie.kind)
        self.assertEqual(True, ie.executable)

    def test_directory_converted_to_submodule(self):
        base_inv = Inventory()
        base_inv.add_path('foo', 'directory')
        base_inv.add_path('foo/bar', 'file')
        othertree = Blob.from_string(b'someotherthing')
        blob = Blob.from_string(b'bar')
        tree = Tree()
        tree.add(b'bar', 57344, blob.id)
        objects = {tree.id: tree}
        ret, child_modes = import_git_submodule(self._texts, self._mapping, b'foo', b'foo', (tree.id, othertree.id), base_inv, base_inv.root.file_id, b'somerevid', [], objects.__getitem__, (stat.S_IFDIR | 493, S_IFGITLINK), DummyStoreUpdater(), self._mapping.generate_file_id)
        self.assertEqual(child_modes, {})
        self.assertEqual(2, len(ret))
        self.assertEqual(ret[0], ('foo/bar', None, base_inv.path2id('foo/bar'), None))
        self.assertEqual(ret[1][:3], ('foo', 'foo', self._mapping.generate_file_id('foo')))
        ie = ret[1][3]
        self.assertEqual(ie.kind, 'tree-reference')