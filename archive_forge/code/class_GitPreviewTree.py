import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
class GitPreviewTree(PreviewTree, GitTree):
    """Partial implementation of Tree to support show_diff_trees"""
    supports_file_ids = False

    def __init__(self, transform):
        PreviewTree.__init__(self, transform)
        self.store = transform._tree.store
        self.mapping = transform._tree.mapping
        self._final_paths = FinalPaths(transform)

    def supports_setting_file_ids(self):
        return False

    def supports_symlinks(self):
        return self._transform._create_symlinks

    def _supports_executable(self):
        return self._transform._limbo_supports_executable()

    def walkdirs(self, prefix=''):
        pending = [self._transform.root]
        while len(pending) > 0:
            parent_id = pending.pop()
            children = []
            subdirs = []
            prefix = prefix.rstrip('/')
            parent_path = self._final_paths.get_path(parent_id)
            for child_id in self._all_children(parent_id):
                path_from_root = self._final_paths.get_path(child_id)
                basename = self._transform.final_name(child_id)
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

    def iter_changes(self, from_tree, include_unchanged=False, specific_files=None, pb=None, extra_trees=None, require_versioned=True, want_unversioned=False):
        """See InterTree.iter_changes.

        This has a fast path that is only used when the from_tree matches
        the transform tree, and no fancy options are supplied.
        """
        return InterTree.get(from_tree, self).iter_changes(include_unchanged=include_unchanged, specific_files=specific_files, pb=pb, extra_trees=extra_trees, require_versioned=require_versioned, want_unversioned=want_unversioned)

    def get_file(self, path):
        """See Tree.get_file"""
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise _mod_transport.NoSuchFile(path)
        if trans_id in self._transform._new_contents:
            name = self._transform._limbo_name(trans_id)
            return open(name, 'rb')
        if trans_id in self._transform._removed_contents:
            raise _mod_transport.NoSuchFile(path)
        orig_path = self._transform.tree_path(trans_id)
        return self._transform._tree.get_file(orig_path)

    def get_symlink_target(self, path):
        """See Tree.get_symlink_target"""
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise _mod_transport.NoSuchFile(path)
        if trans_id not in self._transform._new_contents:
            orig_path = self._transform.tree_path(trans_id)
            return self._transform._tree.get_symlink_target(orig_path)
        name = self._transform._limbo_name(trans_id)
        return osutils.readlink(name)

    def annotate_iter(self, path, default_revision=_mod_revision.CURRENT_REVISION):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            return None
        orig_path = self._transform.tree_path(trans_id)
        if orig_path is not None:
            old_annotation = self._transform._tree.annotate_iter(orig_path, default_revision=default_revision)
        else:
            old_annotation = []
        try:
            lines = self.get_file_lines(path)
        except _mod_transport.NoSuchFile:
            return None
        return annotate.reannotate([old_annotation], lines, default_revision)

    def path2id(self, path):
        if isinstance(path, list):
            if path == []:
                path = ['']
            path = osutils.pathjoin(*path)
        if not self.is_versioned(path):
            return None
        return self._transform._tree.mapping.generate_file_id(path)

    def get_file_text(self, path):
        """Return the byte content of a file.

        :param path: The path of the file.

        :returns: A single byte string for the whole file.
        """
        with self.get_file(path) as my_file:
            return my_file.read()

    def get_file_lines(self, path):
        """Return the content of a file, as lines.

        :param path: The path of the file.
        """
        return osutils.split_lines(self.get_file_text(path))

    def extras(self):
        possible_extras = {self._transform.trans_id_tree_path(p) for p in self._transform._tree.extras()}
        possible_extras.update(self._transform._new_contents)
        possible_extras.update(self._transform._removed_id)
        for trans_id in possible_extras:
            if not self._transform.final_is_versioned(trans_id):
                yield self._final_paths._determine_path(trans_id)

    def path_content_summary(self, path):
        trans_id = self._path2trans_id(path)
        tt = self._transform
        tree_path = tt.tree_path(trans_id)
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

    def get_file_mtime(self, path):
        """See Tree.get_file_mtime"""
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise _mod_transport.NoSuchFile(path)
        if trans_id not in self._transform._new_contents:
            return self._transform._tree.get_file_mtime(self._transform.tree_path(trans_id))
        name = self._transform._limbo_name(trans_id)
        statval = os.lstat(name)
        return statval.st_mtime

    def is_versioned(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            return False
        if trans_id in self._transform._versioned:
            return True
        if trans_id in self._transform._removed_id:
            return False
        orig_path = self._transform.tree_path(trans_id)
        return self._transform._tree.is_versioned(orig_path)

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        if recurse_nested:
            raise NotImplementedError('follow tree references not yet supported')
        for trans_id, path in self._list_files_by_dir():
            entry, is_versioned = self._transform.final_entry(trans_id)
            if entry is None:
                continue
            if not is_versioned and entry.kind != 'directory':
                continue
            if specific_files is not None and path not in specific_files:
                continue
            if entry is not None:
                yield (path, entry)

    def _list_files_by_dir(self):
        todo = [ROOT_PARENT]
        while len(todo) > 0:
            parent = todo.pop()
            children = list(self._all_children(parent))
            paths = dict(zip(children, self._final_paths.get_paths(children)))
            children.sort(key=paths.get)
            todo.extend(reversed(children))
            for trans_id in children:
                yield (trans_id, paths[trans_id][0])

    def revision_tree(self, revision_id):
        return self._transform._tree.revision_tree(revision_id)

    def _stat_limbo_file(self, trans_id):
        name = self._transform._limbo_name(trans_id)
        return os.lstat(name)

    def git_snapshot(self, want_unversioned=False):
        extra = set()
        os = []
        for trans_id, path in self._list_files_by_dir():
            if not self._transform.final_is_versioned(trans_id):
                if not want_unversioned:
                    continue
                extra.add(path)
            o, mode = self._transform.final_git_entry(trans_id)
            if o is not None:
                self.store.add_object(o)
                os.append((encode_git_path(path), o.id, mode))
        if not os:
            return (None, extra)
        return (commit_tree(self.store, os), extra)

    def iter_child_entries(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise _mod_transport.NoSuchFile(path)
        for child_trans_id in self._all_children(trans_id):
            entry, is_versioned = self._transform.final_entry(trans_id)
            if not is_versioned:
                continue
            if entry is not None:
                yield entry