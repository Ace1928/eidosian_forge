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