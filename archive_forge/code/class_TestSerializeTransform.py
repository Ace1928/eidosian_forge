import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
class TestSerializeTransform(tests.TestCaseWithTransport):
    _test_needs_features = [features.UnicodeFilenameFeature]

    def get_preview(self, tree=None):
        if tree is None:
            tree = self.make_branch_and_tree('tree')
        tt = tree.preview_transform()
        self.addCleanup(tt.finalize)
        return tt

    def assertSerializesTo(self, expected, tt):
        records = list(tt.serialize(FakeSerializer()))
        self.assertEqual(expected, records)

    @staticmethod
    def default_attribs():
        return {b'_id_number': 1, b'_new_name': {}, b'_new_parent': {}, b'_new_executability': {}, b'_new_id': {}, b'_tree_path_ids': {b'': b'new-0'}, b'_removed_id': [], b'_removed_contents': [], b'_non_present_ids': {}}

    def make_records(self, attribs, contents):
        records = [(((b'attribs',),), bencode.bencode(attribs))]
        records.extend([(((n, k),), c) for n, k, c in contents])
        return records

    def creation_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 3
        attribs[b'_new_name'] = {b'new-1': 'fooሴ'.encode(), b'new-2': b'qux'}
        attribs[b'_new_id'] = {b'new-1': b'baz', b'new-2': b'quxx'}
        attribs[b'_new_parent'] = {b'new-1': b'new-0', b'new-2': b'new-0'}
        attribs[b'_new_executability'] = {b'new-1': 1}
        contents = [(b'new-1', b'file', b'i 1\nbar\n'), (b'new-2', b'directory', b'')]
        return self.make_records(attribs, contents)

    def test_serialize_creation(self):
        tt = self.get_preview()
        tt.new_file('fooሴ', tt.root, [b'bar'], b'baz', True)
        tt.new_directory('qux', tt.root, b'quxx')
        self.assertSerializesTo(self.creation_records(), tt)

    def test_deserialize_creation(self):
        tt = self.get_preview()
        tt.deserialize(iter(self.creation_records()))
        self.assertEqual(3, tt._id_number)
        self.assertEqual({'new-1': 'fooሴ', 'new-2': 'qux'}, tt._new_name)
        self.assertEqual({'new-1': b'baz', 'new-2': b'quxx'}, tt._new_id)
        self.assertEqual({'new-1': tt.root, 'new-2': tt.root}, tt._new_parent)
        self.assertEqual({b'baz': 'new-1', b'quxx': 'new-2'}, tt._r_new_id)
        self.assertEqual({'new-1': True}, tt._new_executability)
        self.assertEqual({'new-1': 'file', 'new-2': 'directory'}, tt._new_contents)
        with open(tt._limbo_name('new-1'), 'rb') as foo_limbo:
            foo_content = foo_limbo.read()
        self.assertEqual(b'bar', foo_content)

    def symlink_creation_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 2
        attribs[b'_new_name'] = {b'new-1': 'fooሴ'.encode()}
        attribs[b'_new_parent'] = {b'new-1': b'new-0'}
        contents = [(b'new-1', b'symlink', 'barሴ'.encode())]
        return self.make_records(attribs, contents)

    def test_serialize_symlink_creation(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tt = self.get_preview()
        tt.new_symlink('fooሴ', tt.root, 'barሴ')
        self.assertSerializesTo(self.symlink_creation_records(), tt)

    def test_deserialize_symlink_creation(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tt = self.get_preview()
        tt.deserialize(iter(self.symlink_creation_records()))
        abspath = tt._limbo_name('new-1')
        foo_content = osutils.readlink(abspath)
        self.assertEqual('barሴ', foo_content)

    def make_destruction_preview(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['fooሴ', 'bar'])
        tree.add(['fooሴ', 'bar'], ids=[b'foo-id', b'bar-id'])
        return self.get_preview(tree)

    def destruction_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 3
        attribs[b'_removed_id'] = [b'new-1']
        attribs[b'_removed_contents'] = [b'new-2']
        attribs[b'_tree_path_ids'] = {b'': b'new-0', 'fooሴ'.encode(): b'new-1', b'bar': b'new-2'}
        return self.make_records(attribs, [])

    def test_serialize_destruction(self):
        tt = self.make_destruction_preview()
        foo_trans_id = tt.trans_id_tree_path('fooሴ')
        tt.unversion_file(foo_trans_id)
        bar_trans_id = tt.trans_id_tree_path('bar')
        tt.delete_contents(bar_trans_id)
        self.assertSerializesTo(self.destruction_records(), tt)

    def test_deserialize_destruction(self):
        tt = self.make_destruction_preview()
        tt.deserialize(iter(self.destruction_records()))
        self.assertEqual({'fooሴ': 'new-1', 'bar': 'new-2', '': tt.root}, tt._tree_path_ids)
        self.assertEqual({'new-1': 'fooሴ', 'new-2': 'bar', tt.root: ''}, tt._tree_id_paths)
        self.assertEqual({'new-1'}, tt._removed_id)
        self.assertEqual({'new-2'}, tt._removed_contents)

    def missing_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 2
        attribs[b'_non_present_ids'] = {b'boo': b'new-1'}
        return self.make_records(attribs, [])

    def test_serialize_missing(self):
        tt = self.get_preview()
        tt.trans_id_file_id(b'boo')
        self.assertSerializesTo(self.missing_records(), tt)

    def test_deserialize_missing(self):
        tt = self.get_preview()
        tt.deserialize(iter(self.missing_records()))
        self.assertEqual({b'boo': 'new-1'}, tt._non_present_ids)

    def make_modification_preview(self):
        LINES_ONE = b'aa\nbb\ncc\ndd\n'
        LINES_TWO = b'z\nbb\nx\ndd\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', LINES_ONE)])
        tree.add('file', ids=b'file-id')
        return (self.get_preview(tree), [LINES_TWO])

    def modification_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 2
        attribs[b'_tree_path_ids'] = {b'file': b'new-1', b'': b'new-0'}
        attribs[b'_removed_contents'] = [b'new-1']
        contents = [(b'new-1', b'file', b'i 1\nz\n\nc 0 1 1 1\ni 1\nx\n\nc 0 3 3 1\n')]
        return self.make_records(attribs, contents)

    def test_serialize_modification(self):
        tt, LINES = self.make_modification_preview()
        trans_id = tt.trans_id_file_id(b'file-id')
        tt.delete_contents(trans_id)
        tt.create_file(LINES, trans_id)
        self.assertSerializesTo(self.modification_records(), tt)

    def test_deserialize_modification(self):
        tt, LINES = self.make_modification_preview()
        tt.deserialize(iter(self.modification_records()))
        self.assertFileEqual(b''.join(LINES), tt._limbo_name('new-1'))

    def make_kind_change_preview(self):
        LINES = b'a\nb\nc\nd\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo/'])
        tree.add('foo', ids=b'foo-id')
        return (self.get_preview(tree), [LINES])

    def kind_change_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 2
        attribs[b'_tree_path_ids'] = {b'foo': b'new-1', b'': b'new-0'}
        attribs[b'_removed_contents'] = [b'new-1']
        contents = [(b'new-1', b'file', b'i 4\na\nb\nc\nd\n\n')]
        return self.make_records(attribs, contents)

    def test_serialize_kind_change(self):
        tt, LINES = self.make_kind_change_preview()
        trans_id = tt.trans_id_file_id(b'foo-id')
        tt.delete_contents(trans_id)
        tt.create_file(LINES, trans_id)
        self.assertSerializesTo(self.kind_change_records(), tt)

    def test_deserialize_kind_change(self):
        tt, LINES = self.make_kind_change_preview()
        tt.deserialize(iter(self.kind_change_records()))
        self.assertFileEqual(b''.join(LINES), tt._limbo_name('new-1'))

    def make_add_contents_preview(self):
        LINES = b'a\nb\nc\nd\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo')
        os.unlink('tree/foo')
        return (self.get_preview(tree), LINES)

    def add_contents_records(self):
        attribs = self.default_attribs()
        attribs[b'_id_number'] = 2
        attribs[b'_tree_path_ids'] = {b'foo': b'new-1', b'': b'new-0'}
        contents = [(b'new-1', b'file', b'i 4\na\nb\nc\nd\n\n')]
        return self.make_records(attribs, contents)

    def test_serialize_add_contents(self):
        tt, LINES = self.make_add_contents_preview()
        trans_id = tt.trans_id_tree_path('foo')
        tt.create_file([LINES], trans_id)
        self.assertSerializesTo(self.add_contents_records(), tt)

    def test_deserialize_add_contents(self):
        tt, LINES = self.make_add_contents_preview()
        tt.deserialize(iter(self.add_contents_records()))
        self.assertFileEqual(LINES, tt._limbo_name('new-1'))

    def test_get_parents_lines(self):
        LINES_ONE = b'aa\nbb\ncc\ndd\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', LINES_ONE)])
        tree.add('file', ids=b'file-id')
        tt = self.get_preview(tree)
        trans_id = tt.trans_id_tree_path('file')
        self.assertEqual(([b'aa\n', b'bb\n', b'cc\n', b'dd\n'],), tt._get_parents_lines(trans_id))

    def test_get_parents_texts(self):
        LINES_ONE = b'aa\nbb\ncc\ndd\n'
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', LINES_ONE)])
        tree.add('file', ids=b'file-id')
        tt = self.get_preview(tree)
        trans_id = tt.trans_id_tree_path('file')
        self.assertEqual((LINES_ONE,), tt._get_parents_texts(trans_id))