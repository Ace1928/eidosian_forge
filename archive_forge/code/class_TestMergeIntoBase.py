import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
class TestMergeIntoBase(tests.TestCaseWithTransport):

    def setup_simple_branch(self, relpath, shape=None, root_id=None):
        """One commit, containing tree specified by optional shape.

        Default is empty tree (just root entry).
        """
        if root_id is None:
            root_id = b'%s-root-id' % (relpath.encode('ascii'),)
        wt = self.make_branch_and_tree(relpath)
        wt.set_root_id(root_id)
        if shape is not None:
            adjusted_shape = [relpath + '/' + elem for elem in shape]
            self.build_tree(adjusted_shape)
            ids = [b'%s-%s-id' % (relpath.encode('utf-8'), basename(elem.rstrip('/')).encode('ascii')) for elem in shape]
            wt.add(shape, ids=ids)
        rev_id = b'r1-%s' % (relpath.encode('utf-8'),)
        wt.commit('Initial commit of {}'.format(relpath), rev_id=rev_id)
        self.assertEqual(root_id, wt.path2id(''))
        return wt

    def setup_two_branches(self, custom_root_ids=True):
        """Setup 2 branches, one will be a library, the other a project."""
        if custom_root_ids:
            root_id = None
        else:
            root_id = inventory.ROOT_ID
        project_wt = self.setup_simple_branch('project', ['README', 'dir/', 'dir/file.c'], root_id)
        lib_wt = self.setup_simple_branch('lib1', ['README', 'Makefile', 'foo.c'], root_id)
        return (project_wt, lib_wt)

    def do_merge_into(self, location, merge_as):
        """Helper for using MergeIntoMerger.

        :param location: location of directory to merge from, either the
            location of a branch or of a path inside a branch.
        :param merge_as: the path in a tree to add the new directory as.
        :returns: the conflicts from 'do_merge'.
        """
        with contextlib.ExitStack() as stack:
            wt, subdir_relpath = WorkingTree.open_containing(merge_as)
            stack.enter_context(wt.lock_write())
            branch_to_merge, subdir_to_merge = _mod_branch.Branch.open_containing(location)
            stack.enter_context(branch_to_merge.lock_read())
            other_tree = branch_to_merge.basis_tree()
            stack.enter_context(other_tree.lock_read())
            merger = _mod_merge.MergeIntoMerger(this_tree=wt, other_tree=other_tree, other_branch=branch_to_merge, target_subdir=subdir_relpath, source_subpath=subdir_to_merge)
            merger.set_base_revision(_mod_revision.NULL_REVISION, branch_to_merge)
            conflicts = merger.do_merge()
            merger.set_pending()
            return conflicts

    def assertTreeEntriesEqual(self, expected_entries, tree):
        """Assert that 'tree' contains the expected inventory entries.

        :param expected_entries: sequence of (path, file-id) pairs.
        """
        files = [(path, ie.file_id) for path, ie in tree.iter_entries_by_dir()]
        self.assertEqual(expected_entries, files)