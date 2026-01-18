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
def _index_add_entry(self, path, kind, reference_revision=None, symlink_target=None):
    if kind == 'directory':
        return
    elif kind == 'file':
        blob = Blob()
        try:
            file, stat_val = self.get_file_with_stat(path)
        except (_mod_transport.NoSuchFile, OSError):
            file = BytesIO()
            stat_val = os.stat_result((stat.S_IFREG | 420, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        with file:
            blob.set_raw_string(file.read())
        if blob.id not in self.store:
            self.store.add_object(blob)
        hexsha = blob.id
    elif kind == 'symlink':
        blob = Blob()
        try:
            stat_val = self._lstat(path)
        except OSError:
            stat_val = os.stat_result((stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        if symlink_target is None:
            symlink_target = self.get_symlink_target(path)
        blob.set_raw_string(encode_git_path(symlink_target))
        if blob.id not in self.store:
            self.store.add_object(blob)
        hexsha = blob.id
    elif kind == 'tree-reference':
        if reference_revision is not None:
            hexsha = self.branch.lookup_bzr_revision_id(reference_revision)[0]
        else:
            hexsha = self._read_submodule_head(path)
            if hexsha is None:
                raise errors.NoCommits(path)
        try:
            stat_val = self._lstat(path)
        except OSError:
            stat_val = os.stat_result((S_IFGITLINK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        stat_val = os.stat_result((S_IFGITLINK,) + stat_val[1:])
    else:
        raise AssertionError("unknown kind '%s'" % kind)
    ensure_normalized_path(path)
    encoded_path = encode_git_path(path)
    if b'\r' in encoded_path or b'\n' in encoded_path:
        trace.mutter('ignoring path with invalid newline in it: %r', path)
        return
    index, index_path = self._lookup_index(encoded_path)
    index[index_path] = index_entry_from_stat(stat_val, hexsha)
    self._index_dirty = True
    if self._versioned_dirs is not None:
        self._ensure_versioned_dir(index_path)