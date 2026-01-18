import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
class InventoryPreviewTree(PreviewTree, inventorytree.InventoryTree):
    """Partial implementation of Tree to support show_diff_trees"""

    def __init__(self, transform):
        PreviewTree.__init__(self, transform)
        self._final_paths = FinalPaths(transform)
        self._iter_changes_cache = {c.file_id: c for c in self._transform.iter_changes()}

    def supports_setting_file_ids(self):
        return True

    def supports_symlinks(self):
        return self._transform._create_symlinks

    def supports_tree_reference(self):
        return False

    def _content_change(self, file_id):
        """Return True if the content of this file changed"""
        changes = self._iter_changes_cache.get(file_id)
        return changes is not None and changes.changed_content

    def _get_file_revision(self, path, file_id, vf, tree_revision):
        parent_keys = [(file_id, t.get_file_revision(t.id2path(file_id))) for t in self._iter_parent_trees()]
        vf.add_lines((file_id, tree_revision), parent_keys, self.get_file_lines(path))
        repo = self._get_repository()
        base_vf = repo.texts
        if base_vf not in vf.fallback_versionedfiles:
            vf.fallback_versionedfiles.append(base_vf)
        return tree_revision

    def _stat_limbo_file(self, trans_id):
        name = self._transform._limbo_name(trans_id)
        return os.lstat(name)

    def _comparison_data(self, entry, path):
        kind, size, executable, link_or_sha1 = self.path_content_summary(path)
        if kind == 'missing':
            kind = None
            executable = False
        else:
            file_id = self._transform.final_file_id(self._path2trans_id(path))
            executable = self.is_executable(path)
        return (kind, executable, None)

    @property
    def root_inventory(self):
        """This Tree does not use inventory as its backing data."""
        raise NotImplementedError(PreviewTree.root_inventory)

    def all_file_ids(self):
        tree_ids = set(self._transform._tree.all_file_ids())
        tree_ids.difference_update((self._transform.tree_file_id(t) for t in self._transform._removed_id))
        tree_ids.update(self._transform._new_id.values())
        return tree_ids

    def all_versioned_paths(self):
        tree_paths = set(self._transform._tree.all_versioned_paths())
        tree_paths.difference_update((self._transform.trans_id_tree_path(t) for t in self._transform._removed_id))
        tree_paths.update((self._final_paths._determine_path(t) for t in self._transform._new_id))
        return tree_paths

    def path2id(self, path):
        if isinstance(path, list):
            if path == []:
                path = ['']
            path = osutils.pathjoin(*path)
        return self._transform.final_file_id(self._path2trans_id(path))

    def id2path(self, file_id, recurse='down'):
        trans_id = self._transform.trans_id_file_id(file_id)
        try:
            return self._final_paths._determine_path(trans_id)
        except NoFinalPath:
            raise errors.NoSuchId(self, file_id)

    def extras(self):
        possible_extras = {self._transform.trans_id_tree_path(p) for p in self._transform._tree.extras()}
        possible_extras.update(self._transform._new_contents)
        possible_extras.update(self._transform._removed_id)
        for trans_id in possible_extras:
            if self._transform.final_file_id(trans_id) is None:
                yield self._final_paths._determine_path(trans_id)

    def _make_inv_entries(self, ordered_entries, specific_files=None):
        for trans_id, parent_file_id in ordered_entries:
            file_id = self._transform.final_file_id(trans_id)
            if file_id is None:
                continue
            if specific_files is not None and self._final_paths.get_path(trans_id) not in specific_files:
                continue
            kind = self._transform.final_kind(trans_id)
            if kind is None:
                kind = self._transform._tree.stored_kind(self._transform._tree.id2path(file_id))
            new_entry = inventory.make_entry(kind, self._transform.final_name(trans_id), parent_file_id, file_id)
            yield (new_entry, trans_id)

    def _list_files_by_dir(self):
        todo = [ROOT_PARENT]
        ordered_ids = []
        while len(todo) > 0:
            parent = todo.pop()
            parent_file_id = self._transform.final_file_id(parent)
            children = list(self._all_children(parent))
            paths = dict(zip(children, self._final_paths.get_paths(children)))
            children.sort(key=paths.get)
            todo.extend(reversed(children))
            for trans_id in children:
                ordered_ids.append((trans_id, parent_file_id))
        return ordered_ids

    def iter_child_entries(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise _mod_transport.NoSuchFile(path)
        todo = [(child_trans_id, trans_id) for child_trans_id in self._all_children(trans_id)]
        for entry, trans_id in self._make_inv_entries(todo):
            yield entry

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        if recurse_nested:
            raise NotImplementedError('follow tree references not yet supported')
        ordered_ids = self._list_files_by_dir()
        for entry, trans_id in self._make_inv_entries(ordered_ids, specific_files):
            yield (self._final_paths.get_path(trans_id), entry)

    def _iter_entries_for_dir(self, dir_path):
        """Return path, entry for items in a directory without recursing down."""
        ordered_ids = []
        dir_trans_id = self._path2trans_id(dir_path)
        dir_id = self._transform.final_file_id(dir_trans_id)
        for child_trans_id in self._all_children(dir_trans_id):
            ordered_ids.append((child_trans_id, dir_id))
        path_entries = []
        for entry, trans_id in self._make_inv_entries(ordered_ids):
            path_entries.append((self._final_paths.get_path(trans_id), entry))
        path_entries.sort()
        return path_entries

    def list_files(self, include_root=False, from_dir=None, recursive=True, recurse_nested=False):
        """See WorkingTree.list_files."""
        if recurse_nested:
            raise NotImplementedError('follow tree references not yet supported')
        if from_dir == '.':
            from_dir = None
        if recursive:
            prefix = None
            if from_dir:
                prefix = from_dir + '/'
            entries = self.iter_entries_by_dir()
            for path, entry in entries:
                if entry.name == '' and (not include_root):
                    continue
                if prefix:
                    if not path.startswith(prefix):
                        continue
                    path = path[len(prefix):]
                yield (path, 'V', entry.kind, entry)
        else:
            if from_dir is None and include_root is True:
                root_entry = inventory.make_entry('directory', '', ROOT_PARENT, self.path2id(''))
                yield ('', 'V', 'directory', root_entry)
            entries = self._iter_entries_for_dir(from_dir or '')
            for path, entry in entries:
                yield (path, 'V', entry.kind, entry)

    def get_file_mtime(self, path):
        """See Tree.get_file_mtime"""
        file_id = self.path2id(path)
        if file_id is None:
            raise _mod_transport.NoSuchFile(path)
        if not self._content_change(file_id):
            return self._transform._tree.get_file_mtime(self._transform._tree.id2path(file_id))
        trans_id = self._path2trans_id(path)
        return self._stat_limbo_file(trans_id).st_mtime

    def path_content_summary(self, path):
        trans_id = self._path2trans_id(path)
        tt = self._transform
        tree_path = tt._tree_id_paths.get(trans_id)
        kind = tt._new_contents.get(trans_id)
        if kind is None:
            if tree_path is None or trans_id in tt._removed_contents:
                return ('missing', None, None, None)
            summary = tt._tree.path_content_summary(tree_path)
            kind, size, executable, link_or_sha1 = summary
        else:
            link_or_sha1 = None
            limbo_name = tt._limbo_name(trans_id)
            if trans_id in tt._new_reference_revision:
                kind = 'tree-reference'
                link_or_sha1 = tt._new_reference_revision
            if kind == 'file':
                statval = os.lstat(limbo_name)
                size = statval.st_size
                if not tt._limbo_supports_executable():
                    executable = False
                else:
                    executable = statval.st_mode & S_IEXEC
            else:
                size = None
                executable = None
            if kind == 'symlink':
                link_or_sha1 = os.readlink(limbo_name)
                if not isinstance(link_or_sha1, str):
                    link_or_sha1 = os.fsdecode(link_or_sha1)
        executable = tt._new_executability.get(trans_id, executable)
        return (kind, size, executable, link_or_sha1)

    def iter_changes(self, from_tree, include_unchanged=False, specific_files=None, pb=None, extra_trees=None, require_versioned=True, want_unversioned=False):
        """See InterTree.iter_changes.

        This has a fast path that is only used when the from_tree matches
        the transform tree, and no fancy options are supplied.
        """
        if from_tree is not self._transform._tree or include_unchanged or specific_files or want_unversioned:
            return tree.InterTree.get(from_tree, self).iter_changes(include_unchanged=include_unchanged, specific_files=specific_files, pb=pb, extra_trees=extra_trees, require_versioned=require_versioned, want_unversioned=want_unversioned)
        if want_unversioned:
            raise ValueError('want_unversioned is not supported')
        return self._transform.iter_changes()

    def annotate_iter(self, path, default_revision=_mod_revision.CURRENT_REVISION):
        file_id = self.path2id(path)
        changes = self._iter_changes_cache.get(file_id)
        if changes is None:
            if file_id is None:
                old_path = None
            else:
                old_path = self._transform._tree.id2path(file_id)
        else:
            if changes.kind[1] is None:
                return None
            if changes.kind[0] == 'file' and changes.versioned[0]:
                old_path = changes.path[0]
            else:
                old_path = None
        if old_path is not None:
            old_annotation = self._transform._tree.annotate_iter(old_path, default_revision=default_revision)
        else:
            old_annotation = []
        if changes is None:
            if old_path is None:
                return None
            else:
                return old_annotation
        if not changes.changed_content:
            return old_annotation
        return annotate.reannotate([old_annotation], self.get_file_lines(path), default_revision)

    def walkdirs(self, prefix=''):
        pending = [self._transform.root]
        while len(pending) > 0:
            parent_id = pending.pop()
            children = []
            subdirs = []
            prefix = prefix.rstrip('/')
            parent_path = self._final_paths.get_path(parent_id)
            parent_file_id = self._transform.final_file_id(parent_id)
            for child_id in self._all_children(parent_id):
                path_from_root = self._final_paths.get_path(child_id)
                basename = self._transform.final_name(child_id)
                file_id = self._transform.final_file_id(child_id)
                kind = self._transform.final_kind(child_id)
                if kind is not None:
                    versioned_kind = kind
                else:
                    kind = 'unknown'
                    versioned_kind = self._transform._tree.stored_kind(path_from_root)
                if versioned_kind == 'directory':
                    subdirs.append(child_id)
                children.append((path_from_root, basename, kind, None, versioned_kind))
            children.sort()
            if parent_path.startswith(prefix):
                yield (parent_path, children)
            pending.extend(sorted(subdirs, key=self._final_paths.get_path, reverse=True))

    def get_symlink_target(self, path):
        """See Tree.get_symlink_target"""
        file_id = self.path2id(path)
        if not self._content_change(file_id):
            return self._transform._tree.get_symlink_target(path)
        trans_id = self._path2trans_id(path)
        name = self._transform._limbo_name(trans_id)
        return osutils.readlink(name)

    def get_file(self, path):
        """See Tree.get_file"""
        file_id = self.path2id(path)
        if not self._content_change(file_id):
            return self._transform._tree.get_file(path)
        trans_id = self._path2trans_id(path)
        name = self._transform._limbo_name(trans_id)
        return open(name, 'rb')