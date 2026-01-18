import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
class InterDirStateTree(InterInventoryTree):
    """Fast path optimiser for changes_from with dirstate trees.

    This is used only when both trees are in the dirstate working file, and
    the source is any parent within the dirstate, and the destination is
    the current working tree of the same dirstate.
    """

    def __init__(self, source, target):
        super().__init__(source, target)
        if not InterDirStateTree.is_compatible(source, target):
            raise Exception('invalid source %r and target %r' % (source, target))

    @staticmethod
    def make_source_parent_tree(source, target):
        """Change the source tree into a parent of the target."""
        revid = source.commit('record tree')
        target.branch.fetch(source.branch, revid)
        target.set_parent_ids([revid])
        return (target.basis_tree(), target)

    @classmethod
    def make_source_parent_tree_python_dirstate(klass, test_case, source, target):
        result = klass.make_source_parent_tree(source, target)
        result[1]._iter_changes = dirstate.ProcessEntryPython
        return result

    @classmethod
    def make_source_parent_tree_compiled_dirstate(klass, test_case, source, target):
        from .tests.test__dirstate_helpers import compiled_dirstate_helpers_feature
        test_case.requireFeature(compiled_dirstate_helpers_feature)
        from ._dirstate_helpers_pyx import ProcessEntryC
        result = klass.make_source_parent_tree(source, target)
        result[1]._iter_changes = ProcessEntryC
        return result
    _matching_from_tree_format = WorkingTreeFormat4()
    _matching_to_tree_format = WorkingTreeFormat4()

    @classmethod
    def _test_mutable_trees_to_test_trees(klass, test_case, source, target):
        raise NotImplementedError

    def iter_changes(self, include_unchanged=False, specific_files=None, pb=None, extra_trees=[], require_versioned=True, want_unversioned=False):
        """Return the changes from source to target.

        :return: An iterator that yields tuples. See InterTree.iter_changes
            for details.
        :param specific_files: An optional list of file paths to restrict the
            comparison to. When mapping filenames to ids, all matches in all
            trees (including optional extra_trees) are used, and all children of
            matched directories are included.
        :param include_unchanged: An optional boolean requesting the inclusion of
            unchanged entries in the result.
        :param extra_trees: An optional list of additional trees to use when
            mapping the contents of specific_files (paths) to file_ids.
        :param require_versioned: If True, all files in specific_files must be
            versioned in one of source, target, extra_trees or
            PathsNotVersionedError is raised.
        :param want_unversioned: Should unversioned files be returned in the
            output. An unversioned file is defined as one with (False, False)
            for the versioned pair.
        """
        if extra_trees or specific_files == []:
            return super().iter_changes(include_unchanged, specific_files, pb, extra_trees, require_versioned, want_unversioned=want_unversioned)
        parent_ids = self.target.get_parent_ids()
        if not (self.source._revision_id in parent_ids or self.source._revision_id == _mod_revision.NULL_REVISION):
            raise AssertionError('revision {%s} is not stored in {%s}, but %s can only be used for trees stored in the dirstate' % (self.source._revision_id, self.target, self.iter_changes))
        target_index = 0
        if self.source._revision_id == _mod_revision.NULL_REVISION:
            source_index = None
            indices = (target_index,)
        else:
            if not self.source._revision_id in parent_ids:
                raise AssertionError('Failure: source._revision_id: {} not in target.parent_ids({})'.format(self.source._revision_id, parent_ids))
            source_index = 1 + parent_ids.index(self.source._revision_id)
            indices = (source_index, target_index)
        if specific_files is None:
            specific_files = {''}
        state = self.target.current_dirstate()
        state._read_dirblocks_if_needed()
        if require_versioned:
            not_versioned = []
            for path in specific_files:
                path_entries = state._entries_for_path(path.encode('utf-8'))
                if not path_entries:
                    not_versioned.append(path)
                    continue
                found_versioned = False
                for entry in path_entries:
                    for index in indices:
                        if entry[1][index][0] != b'a':
                            found_versioned = True
                            break
                if not found_versioned:
                    not_versioned.append(path)
            if len(not_versioned) > 0:
                raise errors.PathsNotVersionedError(not_versioned)
        search_specific_files_utf8 = set()
        for path in osutils.minimum_path_selection(specific_files):
            search_specific_files_utf8.add(path.encode('utf8'))
        iter_changes = self.target._iter_changes(include_unchanged, self.target._supports_executable(), search_specific_files_utf8, state, source_index, target_index, want_unversioned, self.target)
        return iter_changes.iter_changes()

    @staticmethod
    def is_compatible(source, target):
        if not isinstance(target, DirStateWorkingTree):
            return False
        if not isinstance(source, (revisiontree.RevisionTree, DirStateRevisionTree)):
            return False
        if not (source._revision_id == _mod_revision.NULL_REVISION or source._revision_id in target.get_parent_ids()):
            return False
        return True