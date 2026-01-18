import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class TestGetTreesAndBranchesToDiffLocked(tests.TestCaseWithTransport):

    def call_gtabtd(self, path_list, revision_specs, old_url, new_url):
        """Call get_trees_and_branches_to_diff_locked."""
        exit_stack = contextlib.ExitStack()
        self.addCleanup(exit_stack.close)
        return diff.get_trees_and_branches_to_diff_locked(path_list, revision_specs, old_url, new_url, exit_stack)

    def test_basic(self):
        tree = self.make_branch_and_tree('tree')
        old_tree, new_tree, old_branch, new_branch, specific_files, extra_trees = self.call_gtabtd(['tree'], None, None, None)
        self.assertIsInstance(old_tree, revisiontree.RevisionTree)
        self.assertEqual(_mod_revision.NULL_REVISION, old_tree.get_revision_id())
        self.assertEqual(tree.basedir, new_tree.basedir)
        self.assertEqual(tree.branch.base, old_branch.base)
        self.assertEqual(tree.branch.base, new_branch.base)
        self.assertIs(None, specific_files)
        self.assertIs(None, extra_trees)

    def test_with_rev_specs(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'oldcontent')])
        tree.add('file', ids=b'file-id')
        tree.commit('old tree', timestamp=0, rev_id=b'old-id')
        self.build_tree_contents([('tree/file', b'newcontent')])
        tree.commit('new tree', timestamp=0, rev_id=b'new-id')
        revisions = [revisionspec.RevisionSpec.from_string('1'), revisionspec.RevisionSpec.from_string('2')]
        old_tree, new_tree, old_branch, new_branch, specific_files, extra_trees = self.call_gtabtd(['tree'], revisions, None, None)
        self.assertIsInstance(old_tree, revisiontree.RevisionTree)
        self.assertEqual(b'old-id', old_tree.get_revision_id())
        self.assertIsInstance(new_tree, revisiontree.RevisionTree)
        self.assertEqual(b'new-id', new_tree.get_revision_id())
        self.assertEqual(tree.branch.base, old_branch.base)
        self.assertEqual(tree.branch.base, new_branch.base)
        self.assertIs(None, specific_files)
        self.assertEqual(tree.basedir, extra_trees[0].basedir)