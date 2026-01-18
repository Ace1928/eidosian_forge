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
def _get_file_ie(self, name: str, path: str, value: Union[IndexEntry, ConflictedIndexEntry], parent_id) -> Union[GitTreeSymlink, GitTreeDirectory, GitTreeFile, GitTreeSubmodule]:
    if not isinstance(name, str):
        raise TypeError(name)
    if not isinstance(path, str):
        raise TypeError(path)
    if isinstance(value, IndexEntry):
        mode = value.mode
        sha = value.sha
        size = value.size
    elif isinstance(value, ConflictedIndexEntry):
        if value.this is None:
            raise _mod_transport.NoSuchFile(path)
        mode = value.this.mode
        sha = value.this.sha
        size = value.this.size
    else:
        raise TypeError(value)
    file_id = self.path2id(path)
    if not isinstance(file_id, bytes):
        raise TypeError(file_id)
    kind = mode_kind(mode)
    ie = entry_factory[kind](file_id, name, parent_id, git_sha1=sha)
    if kind == 'symlink':
        ie.symlink_target = self.get_symlink_target(path)
    elif kind == 'tree-reference':
        ie.reference_revision = self.get_reference_revision(path)
    elif kind == 'directory':
        pass
    else:
        ie.git_sha1 = sha
        ie.text_size = size
        ie.executable = bool(stat.S_ISREG(mode) and stat.S_IEXEC & mode)
    return ie