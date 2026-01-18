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
class _SmartAddHelper:
    """Helper for MutableTree.smart_add."""

    def get_inventory_delta(self):
        return list(self._invdelta.values())

    def _get_ie(self, inv_path):
        """Retrieve the most up to date inventory entry for a path.

        :param inv_path: Normalized inventory path
        :return: Inventory entry (with possibly invalid .children for
            directories)
        """
        entry = self._invdelta.get(inv_path)
        if entry is not None:
            return entry[3]
        inv_path = self.tree._fix_case_of_inventory_path(inv_path)
        try:
            return next(self.tree.iter_entries_by_dir(specific_files=[inv_path]))[1]
        except StopIteration:
            return None

    def _convert_to_directory(self, this_ie, inv_path):
        """Convert an entry to a directory.

        :param this_ie: Inventory entry
        :param inv_path: Normalized path for the inventory entry
        :return: The new inventory entry
        """
        this_ie = _mod_inventory.InventoryDirectory(this_ie.file_id, this_ie.name, this_ie.parent_id)
        self._invdelta[inv_path] = (inv_path, inv_path, this_ie.file_id, this_ie)
        return this_ie

    def _add_one_and_parent(self, parent_ie, path, kind, inv_path):
        """Add a new entry to the inventory and automatically add unversioned parents.

        :param parent_ie: Parent inventory entry if known, or None.  If
            None, the parent is looked up by name and used if present, otherwise it
            is recursively added.
        :param path: Filesystem path to add
        :param kind: Kind of new entry (file, directory, etc)
        :param inv_path: Inventory path
        :return: Inventory entry for path and a list of paths which have been added.
        """
        inv_dirname = osutils.dirname(inv_path)
        dirname, basename = osutils.split(path)
        if parent_ie is None:
            this_ie = self._get_ie(inv_path)
            if this_ie is not None:
                return this_ie
            parent_ie = self._add_one_and_parent(None, dirname, 'directory', inv_dirname)
        if parent_ie.kind != 'directory':
            parent_ie = self._convert_to_directory(parent_ie, inv_dirname)
        file_id = self.action(self.tree, parent_ie, path, kind)
        entry = _mod_inventory.make_entry(kind, basename, parent_ie.file_id, file_id=file_id)
        self._invdelta[inv_path] = (None, inv_path, entry.file_id, entry)
        self.added.append(inv_path)
        return entry

    def _gather_dirs_to_add(self, user_dirs):
        prev_dir = None
        is_inside = osutils.is_inside_or_parent_of_any
        for path in sorted(user_dirs):
            if prev_dir is None or not is_inside([prev_dir], path):
                inv_path, this_ie = user_dirs[path]
                yield (path, inv_path, this_ie, None)
            prev_dir = path

    def __init__(self, tree, action, conflicts_related=None):
        self.tree = tree
        if action is None:
            self.action = add.AddAction()
        else:
            self.action = action
        self._invdelta = {}
        self.added = []
        self.ignored = {}
        if conflicts_related is None:
            self.conflicts_related = frozenset()
        else:
            self.conflicts_related = conflicts_related

    def add(self, file_list, recurse=True):
        if not file_list:
            file_list = ['.']
        if self.tree.supports_symlinks():
            file_list = list(map(osutils.normalizepath, file_list))
        user_dirs = {}
        for filepath in osutils.canonical_relpaths(self.tree.basedir, file_list):
            if self.tree.is_control_filename(filepath):
                raise errors.ForbiddenControlFileError(filename=filepath)
            abspath = self.tree.abspath(filepath)
            kind = osutils.file_kind(abspath)
            inv_path, _ = osutils.normalized_filename(filepath)
            this_ie = self._get_ie(inv_path)
            if this_ie is None:
                this_ie = self._add_one_and_parent(None, filepath, kind, inv_path)
            if kind == 'directory':
                user_dirs[filepath] = (inv_path, this_ie)
        if not recurse:
            return
        things_to_add = list(self._gather_dirs_to_add(user_dirs))
        illegalpath_re = re.compile('[\\r\\n]')
        for directory, inv_path, this_ie, parent_ie in things_to_add:
            abspath = self.tree.abspath(directory)
            stat_value = None
            if this_ie is None:
                stat_value = osutils.file_stat(abspath)
                kind = osutils.file_kind_from_stat_mode(stat_value.st_mode)
            else:
                kind = this_ie.kind
            if self.action.skip_file(self.tree, abspath, kind, stat_value):
                continue
            if not _mod_inventory.InventoryEntry.versionable_kind(kind):
                trace.warning("skipping %s (can't add file of kind '%s')", abspath, kind)
                continue
            if illegalpath_re.search(directory):
                trace.warning('skipping %r (contains \\n or \\r)' % abspath)
                continue
            if directory in self.conflicts_related:
                trace.warning('skipping %s (generated to help resolve conflicts)', abspath)
                continue
            if kind == 'directory' and directory != '':
                try:
                    transport = _mod_transport.get_transport_from_path(abspath)
                    controldir.ControlDirFormat.find_format(transport)
                    sub_tree = True
                except errors.NotBranchError:
                    sub_tree = False
                except errors.UnsupportedFormatError:
                    sub_tree = True
            else:
                sub_tree = False
            if this_ie is not None:
                pass
            elif sub_tree:
                trace.warning('skipping nested tree %r', abspath)
            else:
                this_ie = self._add_one_and_parent(parent_ie, directory, kind, inv_path)
            if kind == 'directory' and (not sub_tree):
                if this_ie.kind != 'directory':
                    this_ie = self._convert_to_directory(this_ie, inv_path)
                for subf in sorted(os.listdir(abspath)):
                    inv_f, _ = osutils.normalized_filename(subf)
                    subp = osutils.pathjoin(directory, subf)
                    if self.tree.is_control_filename(subp):
                        trace.mutter('skip control directory %r', subp)
                        continue
                    sub_invp = osutils.pathjoin(inv_path, inv_f)
                    entry = self._invdelta.get(sub_invp)
                    if entry is not None:
                        sub_ie = entry[3]
                    else:
                        sub_ie = this_ie.children.get(inv_f)
                    if sub_ie is not None:
                        things_to_add.append((subp, sub_invp, sub_ie, this_ie))
                    else:
                        ignore_glob = self.tree.is_ignored(subp)
                        if ignore_glob is not None:
                            self.ignored.setdefault(ignore_glob, []).append(subp)
                        else:
                            things_to_add.append((subp, sub_invp, None, this_ie))