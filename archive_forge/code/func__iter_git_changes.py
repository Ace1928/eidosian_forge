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
def _iter_git_changes(self, want_unchanged=False, specific_files=None, require_versioned=False, extra_trees=None, want_unversioned=False, include_trees=True):
    trees = [self.source]
    if extra_trees is not None:
        trees.extend(extra_trees)
    if specific_files is not None:
        specific_files = self.target.find_related_paths_across_trees(specific_files, trees, require_versioned=require_versioned)
    with self.lock_read():
        from_tree_sha, from_extras = self.source.git_snapshot(want_unversioned=want_unversioned)
        to_tree_sha, to_extras = self.target.git_snapshot(want_unversioned=want_unversioned)
        changes = tree_changes(self.store, from_tree_sha, to_tree_sha, include_trees=include_trees, rename_detector=self.rename_detector, want_unchanged=want_unchanged, change_type_same=True)
        return (changes, from_extras, to_extras)