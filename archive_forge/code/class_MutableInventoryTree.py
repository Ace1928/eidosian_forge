import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
class MutableInventoryTree(MutableTree, InventoryTree):

    def apply_inventory_delta(self, changes):
        """Apply changes to the inventory as an atomic operation.

        :param changes: An inventory delta to apply to the working tree's
            inventory.
        :return None:
        :seealso Inventory.apply_delta: For details on the changes parameter.
        """
        with self.lock_tree_write():
            self.flush()
            inv = self.root_inventory
            inv.apply_delta(changes)
            self._write_inventory(inv)

    def has_changes(self, _from_tree=None):
        """Quickly check that the tree contains at least one commitable change.

        :param _from_tree: tree to compare against to find changes (default to
            the basis tree and is intended to be used by tests).

        :return: True if a change is found. False otherwise
        """
        with self.lock_read():
            if len(self.get_parent_ids()) > 1:
                return True
            if _from_tree is None:
                _from_tree = self.basis_tree()
            changes = self.iter_changes(_from_tree)
            if self.supports_symlinks():
                try:
                    change = next(changes)
                    if change.parent_id == (None, None):
                        change = next(changes)
                    return True
                except StopIteration:
                    return False
            else:
                changes = filter(lambda c: c[6][0] != 'symlink' and c[4] != (None, None), changes)
                try:
                    next(iter(changes))
                except StopIteration:
                    return False
                return True

    def _fix_case_of_inventory_path(self, path):
        """If our tree isn't case sensitive, return the canonical path"""
        if not self.case_sensitive:
            path = self.get_canonical_path(path)
        return path

    def smart_add(self, file_list, recurse=True, action=None, save=True):
        """Version file_list, optionally recursing into directories.

        This is designed more towards DWIM for humans than API clarity.
        For the specific behaviour see the help for cmd_add().

        :param file_list: List of zero or more paths.  *NB: these are
            interpreted relative to the process cwd, not relative to the
            tree.*  (Add and most other tree methods use tree-relative
            paths.)
        :param action: A reporter to be called with the inventory, parent_ie,
            path and kind of the path being added. It may return a file_id if
            a specific one should be used.
        :param save: Save the inventory after completing the adds. If False
            this provides dry-run functionality by doing the add and not saving
            the inventory.
        :return: A tuple - files_added, ignored_files. files_added is the count
            of added files, and ignored_files is a dict mapping files that were
            ignored to the rule that caused them to be ignored.
        """
        with self.lock_tree_write():
            if getattr(self, 'conflicts', None) is not None:
                conflicts_related = set()
                for c in self.conflicts():
                    conflicts_related.update(c.associated_filenames())
            else:
                conflicts_related = None
            adder = _SmartAddHelper(self, action, conflicts_related)
            adder.add(file_list, recurse=recurse)
            if save:
                invdelta = adder.get_inventory_delta()
                self.apply_inventory_delta(invdelta)
            return (adder.added, adder.ignored)

    def update_basis_by_delta(self, new_revid, delta):
        """Update the parents of this tree after a commit.

        This gives the tree one parent, with revision id new_revid. The
        inventory delta is applied to the current basis tree to generate the
        inventory for the parent new_revid, and all other parent trees are
        discarded.

        All the changes in the delta should be changes synchronising the basis
        tree with some or all of the working tree, with a change to a directory
        requiring that its contents have been recursively included. That is,
        this is not a general purpose tree modification routine, but a helper
        for commit which is not required to handle situations that do not arise
        outside of commit.

        See the inventory developers documentation for the theory behind
        inventory deltas.

        :param new_revid: The new revision id for the trees parent.
        :param delta: An inventory delta (see apply_inventory_delta) describing
            the changes from the current left most parent revision to new_revid.
        """
        if self.last_revision() == new_revid:
            self.set_parent_ids([new_revid])
            return
        basis = self.basis_tree()
        with basis.lock_read():
            inventory = _mod_inventory.mutable_inventory_from_tree(basis)
        inventory.apply_delta(delta)
        rev_tree = InventoryRevisionTree(self.branch.repository, inventory, new_revid)
        self.set_parent_trees([(new_revid, rev_tree)])

    def transform(self, pb=None):
        from .transform import InventoryTreeTransform
        return InventoryTreeTransform(self, pb=pb)

    def add(self, files, kinds=None, ids=None):
        """Add paths to the set of versioned paths.

        Note that the command line normally calls smart_add instead,
        which can automatically recurse.

        This adds the files to the tree, so that they will be
        recorded by the next commit.

        Args:
          files: List of paths to add, relative to the base of the tree.
          kinds: Optional parameter to specify the kinds to be used for
            each file.
          ids: If set, use these instead of automatically generated ids.
            Must be the same length as the list of files, but may
            contain None for ids that are to be autogenerated.

        TODO: Perhaps callback with the ids and paths as they're added.
        """
        if isinstance(files, str):
            if not (ids is None or isinstance(ids, bytes)):
                raise AssertionError()
            if not (kinds is None or isinstance(kinds, str)):
                raise AssertionError()
            files = [files]
            if ids is not None:
                ids = [ids]
            if kinds is not None:
                kinds = [kinds]
        files = [path.strip('/') for path in files]
        if ids is None:
            ids = [None] * len(files)
        elif not len(ids) == len(files):
            raise AssertionError()
        if kinds is None:
            kinds = [None] * len(files)
        elif not len(kinds) == len(files):
            raise AssertionError()
        with self.lock_tree_write():
            for f in files:
                if self.is_control_filename(f):
                    raise errors.ForbiddenControlFileError(filename=f)
                fp = osutils.splitpath(f)
            self._gather_kinds(files, kinds)
            self._add(files, kinds, ids)

    def _gather_kinds(self, files, kinds):
        """Helper function for add - sets the entries of kinds."""
        raise NotImplementedError(self._gather_kinds)

    def _add(self, files, kinds, ids):
        """Helper function for add - updates the inventory.

        :param files: sequence of pathnames, relative to the tree root
        :param kinds: sequence of  inventory kinds of the files (i.e. may
            contain "tree-reference")
        :param ids: sequence of suggested ids for the files (may be None)
        """
        raise NotImplementedError(self._add)