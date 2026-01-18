import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
class TestWorkingTreeIterEntriesByDir_wSubtrees(TestCaseWithTransport):

    def make_simple_tree(self):
        tree = self.make_branch_and_tree('tree', format='development-subtree')
        self.build_tree(['tree/a/', 'tree/a/b/', 'tree/a/b/c'])
        tree.set_root_id(b'root-id')
        tree.add(['a', 'a/b', 'a/b/c'], ids=[b'a-id', b'b-id', b'c-id'])
        tree.commit('initial')
        return tree

    def test_just_directory(self):
        tree = self.make_simple_tree()
        self.assertEqual([('directory', b'root-id'), ('directory', b'a-id'), ('directory', b'b-id'), ('file', b'c-id')], [(ie.kind, ie.file_id) for path, ie in tree.iter_entries_by_dir()])
        self.make_branch_and_tree('tree/a/b')
        self.assertEqual([('tree-reference', b'b-id')], [(ie.kind, ie.file_id) for path, ie in tree.iter_entries_by_dir(specific_files=['a/b'])])

    def test_direct_subtree(self):
        tree = self.make_simple_tree()
        self.make_branch_and_tree('tree/a/b')
        self.assertEqual([('directory', b'root-id'), ('directory', b'a-id'), ('tree-reference', b'b-id')], [(ie.kind, ie.file_id) for path, ie in tree.iter_entries_by_dir()])

    def test_indirect_subtree(self):
        tree = self.make_simple_tree()
        self.make_branch_and_tree('tree/a')
        self.assertEqual([('directory', b'root-id'), ('tree-reference', b'a-id')], [(ie.kind, ie.file_id) for path, ie in tree.iter_entries_by_dir()])