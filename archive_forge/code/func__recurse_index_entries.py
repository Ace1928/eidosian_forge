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
def _recurse_index_entries(self, index=None, basepath=b'', recurse_nested=False):
    with self.lock_read():
        if index is None:
            index = self.index
        for path, value in index.items():
            if isinstance(value, ConflictedIndexEntry):
                if value.this is None:
                    continue
                mode = value.this.mode
            else:
                mode = value.mode
            if S_ISGITLINK(mode) and recurse_nested:
                subindex = self._get_submodule_index(path)
                yield from self._recurse_index_entries(index=subindex, basepath=path, recurse_nested=recurse_nested)
            else:
                yield (posixpath.join(basepath, path), value)