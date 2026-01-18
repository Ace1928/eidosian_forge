import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
class GitRevisionTree(revisiontree.RevisionTree, GitTree):
    """Revision tree implementation based on Git objects."""

    def __init__(self, repository, revision_id):
        self._revision_id = revision_id
        self._repository = repository
        self._submodules = None
        self.store = repository._git.object_store
        if not isinstance(revision_id, bytes):
            raise TypeError(revision_id)
        self.commit_id, self.mapping = repository.lookup_bzr_revision_id(revision_id)
        if revision_id == NULL_REVISION:
            self.tree = None
            self.mapping = default_mapping
        else:
            try:
                commit = self.store[self.commit_id]
            except KeyError:
                raise errors.NoSuchRevision(repository, revision_id)
            self.tree = commit.tree

    def git_snapshot(self, want_unversioned=False):
        return (self.tree, set())

    def _get_submodule_repository(self, relpath):
        if not isinstance(relpath, bytes):
            raise TypeError(relpath)
        try:
            url, section = self._submodule_info()[relpath]
        except KeyError:
            nested_repo_transport = None
        else:
            nested_repo_transport = self._repository.controldir.control_transport.clone(posixpath.join('modules', decode_git_path(section)))
            if not nested_repo_transport.has('.'):
                nested_url = urlutils.join(self._repository.controldir.user_url, decode_git_path(url))
                nested_repo_transport = get_transport(nested_url)
        if nested_repo_transport is None:
            nested_repo_transport = self._repository.controldir.user_transport.clone(decode_git_path(relpath))
        else:
            nested_repo_transport = self._repository.controldir.control_transport.clone(posixpath.join('modules', decode_git_path(section)))
            if not nested_repo_transport.has('.'):
                nested_repo_transport = self._repository.controldir.user_transport.clone(posixpath.join(decode_git_path(section), '.git'))
        try:
            nested_controldir = _mod_controldir.ControlDir.open_from_transport(nested_repo_transport)
        except errors.NotBranchError as e:
            raise MissingNestedTree(decode_git_path(relpath)) from e
        return nested_controldir.find_repository()

    def _get_submodule_store(self, relpath):
        repo = self._get_submodule_repository(relpath)
        if not hasattr(repo, '_git'):
            raise RemoteNestedTree(relpath)
        return repo._git.object_store

    def get_nested_tree(self, path):
        encoded_path = encode_git_path(path)
        nested_repo = self._get_submodule_repository(encoded_path)
        ref_rev = self.get_reference_revision(path)
        return nested_repo.revision_tree(ref_rev)

    def supports_rename_tracking(self):
        return False

    def get_file_revision(self, path):
        change_scanner = self._repository._file_change_scanner
        if self.commit_id == ZERO_SHA:
            return NULL_REVISION
        store, unused_path, commit_id = change_scanner.find_last_change_revision(encode_git_path(path), self.commit_id)
        return self.mapping.revision_id_foreign_to_bzr(commit_id)

    def get_file_mtime(self, path):
        change_scanner = self._repository._file_change_scanner
        if self.commit_id == ZERO_SHA:
            return NULL_REVISION
        try:
            store, unused_path, commit_id = change_scanner.find_last_change_revision(encode_git_path(path), self.commit_id)
        except KeyError:
            raise _mod_transport.NoSuchFile(path)
        commit = store[commit_id]
        return commit.commit_time

    def is_versioned(self, path):
        return self.has_filename(path)

    def path2id(self, path):
        if self.mapping.is_special_file(path):
            return None
        if not self.is_versioned(path):
            return None
        return self.mapping.generate_file_id(osutils.safe_unicode(path))

    def all_versioned_paths(self):
        ret = {''}
        todo = [(self.store, b'', self.tree)]
        while todo:
            store, path, tree_id = todo.pop()
            if tree_id is None:
                continue
            tree = store[tree_id]
            for name, mode, hexsha in tree.items():
                subpath = posixpath.join(path, name)
                ret.add(decode_git_path(subpath))
                if stat.S_ISDIR(mode):
                    todo.append((store, subpath, hexsha))
        return ret

    def _lookup_path(self, path):
        if self.tree is None:
            raise _mod_transport.NoSuchFile(path)
        encoded_path = encode_git_path(path)
        parts = encoded_path.split(b'/')
        hexsha = self.tree
        store = self.store
        mode = None
        for i, p in enumerate(parts):
            if not p:
                continue
            obj = store[hexsha]
            if not isinstance(obj, Tree):
                raise NotTreeError(hexsha)
            try:
                mode, hexsha = obj[p]
            except KeyError:
                raise _mod_transport.NoSuchFile(path)
            if S_ISGITLINK(mode) and i != len(parts) - 1:
                store = self._get_submodule_store(b'/'.join(parts[:i + 1]))
                hexsha = store[hexsha].tree
        return (store, mode, hexsha)

    def is_executable(self, path):
        store, mode, hexsha = self._lookup_path(path)
        if mode is None:
            return False
        return mode_is_executable(mode)

    def kind(self, path):
        store, mode, hexsha = self._lookup_path(path)
        if mode is None:
            return 'directory'
        return mode_kind(mode)

    def has_filename(self, path):
        try:
            self._lookup_path(path)
        except _mod_transport.NoSuchFile:
            return False
        else:
            return True

    def list_files(self, include_root=False, from_dir=None, recursive=True, recurse_nested=False):
        if self.tree is None:
            return
        if from_dir is None or from_dir == '.':
            from_dir = ''
        store, mode, hexsha = self._lookup_path(from_dir)
        if mode is None:
            root_ie = self._get_dir_ie(b'', None)
        else:
            parent_path = posixpath.dirname(from_dir)
            parent_id = self.mapping.generate_file_id(parent_path)
            if mode_kind(mode) == 'directory':
                root_ie = self._get_dir_ie(encode_git_path(from_dir), parent_id)
            else:
                root_ie = self._get_file_ie(store, encode_git_path(from_dir), posixpath.basename(from_dir), mode, hexsha)
        if include_root:
            yield (from_dir, 'V', root_ie.kind, root_ie)
        todo = []
        if root_ie.kind == 'directory':
            todo.append((store, encode_git_path(from_dir), b'', hexsha, root_ie.file_id))
        while todo:
            store, path, relpath, hexsha, parent_id = todo.pop()
            tree = store[hexsha]
            for name, mode, hexsha in tree.iteritems():
                if self.mapping.is_special_file(name):
                    continue
                child_path = posixpath.join(path, name)
                child_relpath = posixpath.join(relpath, name)
                if S_ISGITLINK(mode) and recurse_nested:
                    mode = stat.S_IFDIR
                    store = self._get_submodule_store(child_relpath)
                    hexsha = store[hexsha].tree
                if stat.S_ISDIR(mode):
                    ie = self._get_dir_ie(child_path, parent_id)
                    if recursive:
                        todo.append((store, child_path, child_relpath, hexsha, ie.file_id))
                else:
                    ie = self._get_file_ie(store, child_path, name, mode, hexsha, parent_id)
                yield (decode_git_path(child_relpath), 'V', ie.kind, ie)

    def _get_file_ie(self, store, path: str, name: str, mode: int, hexsha: bytes, parent_id):
        if not isinstance(path, bytes):
            raise TypeError(path)
        if not isinstance(name, bytes):
            raise TypeError(name)
        kind = mode_kind(mode)
        path = decode_git_path(path)
        name = decode_git_path(name)
        file_id = self.mapping.generate_file_id(path)
        ie = entry_factory[kind](file_id, name, parent_id, git_sha1=hexsha)
        if kind == 'symlink':
            ie.symlink_target = decode_git_path(store[hexsha].data)
        elif kind == 'tree-reference':
            ie.reference_revision = self.mapping.revision_id_foreign_to_bzr(hexsha)
        else:
            ie.git_sha1 = hexsha
            ie.text_size = None
            ie.executable = mode_is_executable(mode)
        return ie

    def _get_dir_ie(self, path, parent_id) -> GitTreeDirectory:
        path = decode_git_path(path)
        file_id = self.mapping.generate_file_id(path)
        return GitTreeDirectory(file_id, posixpath.basename(path), parent_id)

    def iter_child_entries(self, path: str):
        store, mode, tree_sha = self._lookup_path(path)
        if mode is not None and (not stat.S_ISDIR(mode)):
            return
        encoded_path = encode_git_path(path)
        file_id = self.path2id(path)
        tree = store[tree_sha]
        for name, mode, hexsha in tree.iteritems():
            if self.mapping.is_special_file(name):
                continue
            child_path = posixpath.join(encoded_path, name)
            if stat.S_ISDIR(mode):
                yield self._get_dir_ie(child_path, file_id)
            else:
                yield self._get_file_ie(store, child_path, name, mode, hexsha, file_id)

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        if self.tree is None:
            return
        if specific_files is not None:
            if specific_files in ([''], []):
                specific_files = None
            else:
                specific_files = {encode_git_path(p) for p in specific_files}
        todo = deque([(self.store, b'', self.tree, self.path2id(''))])
        if specific_files is None or '' in specific_files:
            yield ('', self._get_dir_ie(b'', None))
        while todo:
            store, path, tree_sha, parent_id = todo.popleft()
            tree = store[tree_sha]
            extradirs = []
            for name, mode, hexsha in tree.iteritems():
                if self.mapping.is_special_file(name):
                    continue
                child_path = posixpath.join(path, name)
                child_path_decoded = decode_git_path(child_path)
                if recurse_nested and S_ISGITLINK(mode):
                    try:
                        substore = self._get_submodule_store(child_path)
                    except errors.NotBranchError:
                        substore = store
                    else:
                        mode = stat.S_IFDIR
                        hexsha = substore[hexsha].tree
                else:
                    substore = store
                if stat.S_ISDIR(mode):
                    if specific_files is None or any([p for p in specific_files if p.startswith(child_path)]):
                        extradirs.append((substore, child_path, hexsha, self.path2id(child_path_decoded)))
                if specific_files is None or child_path in specific_files:
                    if stat.S_ISDIR(mode):
                        yield (child_path_decoded, self._get_dir_ie(child_path, parent_id))
                    else:
                        yield (child_path_decoded, self._get_file_ie(substore, child_path, name, mode, hexsha, parent_id))
            todo.extendleft(reversed(extradirs))

    def iter_references(self):
        if self.supports_tree_reference():
            for path, entry in self.iter_entries_by_dir():
                if entry.kind == 'tree-reference':
                    yield path

    def get_revision_id(self):
        """See RevisionTree.get_revision_id."""
        return self._revision_id

    def get_file_sha1(self, path, stat_value=None):
        if self.tree is None:
            raise _mod_transport.NoSuchFile(path)
        return osutils.sha_string(self.get_file_text(path))

    def get_file_verifier(self, path, stat_value=None):
        store, mode, hexsha = self._lookup_path(path)
        return ('GIT', hexsha)

    def get_file_size(self, path):
        store, mode, hexsha = self._lookup_path(path)
        if stat.S_ISREG(mode):
            return len(store[hexsha].data)
        return None

    def get_file_text(self, path):
        """See RevisionTree.get_file_text."""
        store, mode, hexsha = self._lookup_path(path)
        if stat.S_ISREG(mode):
            return store[hexsha].data
        else:
            return b''

    def get_symlink_target(self, path):
        """See RevisionTree.get_symlink_target."""
        store, mode, hexsha = self._lookup_path(path)
        if stat.S_ISLNK(mode):
            return decode_git_path(store[hexsha].data)
        else:
            return None

    def get_reference_revision(self, path):
        """See RevisionTree.get_symlink_target."""
        store, mode, hexsha = self._lookup_path(path)
        if S_ISGITLINK(mode):
            try:
                nested_repo = self._get_submodule_repository(encode_git_path(path))
            except MissingNestedTree:
                return self.mapping.revision_id_foreign_to_bzr(hexsha)
            else:
                try:
                    return nested_repo.lookup_foreign_revision_id(hexsha)
                except KeyError:
                    return self.mapping.revision_id_foreign_to_bzr(hexsha)
        else:
            return None

    def _comparison_data(self, entry, path):
        if entry is None:
            return (None, False, None)
        return (entry.kind, entry.executable, None)

    def path_content_summary(self, path):
        """See Tree.path_content_summary."""
        try:
            store, mode, hexsha = self._lookup_path(path)
        except _mod_transport.NoSuchFile:
            return ('missing', None, None, None)
        kind = mode_kind(mode)
        if kind == 'file':
            executable = mode_is_executable(mode)
            contents = store[hexsha].data
            return (kind, len(contents), executable, osutils.sha_string(contents))
        elif kind == 'symlink':
            return (kind, None, None, decode_git_path(store[hexsha].data))
        elif kind == 'tree-reference':
            nested_repo = self._get_submodule_repository(encode_git_path(path))
            return (kind, None, None, nested_repo.lookup_foreign_revision_id(hexsha))
        else:
            return (kind, None, None, None)

    def _iter_tree_contents(self, include_trees=False):
        if self.tree is None:
            return iter([])
        return iter_tree_contents(self.store, self.tree, include_trees=include_trees)

    def annotate_iter(self, path, default_revision=CURRENT_REVISION):
        """Return an iterator of revision_id, line tuples.

        For working trees (and mutable trees in general), the special
        revision_id 'current:' will be used for lines that are new in this
        tree, e.g. uncommitted changes.
        :param default_revision: For lines that don't match a basis, mark them
            with this revision id. Not all implementations will make use of
            this value.
        """
        with self.lock_read():
            from breezy.annotate import Annotator
            from .annotate import AnnotateProvider
            annotator = Annotator(AnnotateProvider(self._repository._file_change_scanner))
            this_key = (path, self.get_file_revision(path))
            annotations = [(key[-1], line) for key, line in annotator.annotate_flat(this_key)]
            return annotations

    def _get_rules_searcher(self, default_searcher):
        return default_searcher

    def walkdirs(self, prefix=''):
        store, mode, hexsha = self._lookup_path(prefix)
        todo = deque([(store, encode_git_path(prefix), hexsha)])
        while todo:
            store, path, tree_sha = todo.popleft()
            path_decoded = decode_git_path(path)
            tree = store[tree_sha]
            children = []
            for name, mode, hexsha in tree.iteritems():
                if self.mapping.is_special_file(name):
                    continue
                child_path = posixpath.join(path, name)
                if stat.S_ISDIR(mode):
                    todo.append((store, child_path, hexsha))
                children.append((decode_git_path(child_path), decode_git_path(name), mode_kind(mode), None, mode_kind(mode)))
            yield (path_decoded, children)