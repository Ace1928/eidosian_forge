import errno
import itertools
import os
import posixpath
import re
import stat
import sys
from collections import defaultdict
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.file import FileLocked, GitFile
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (ConflictedIndexEntry, Index, IndexEntry, SHA1Writer,
from dulwich.object_store import iter_tree_contents
from dulwich.objects import S_ISGITLINK
from .. import branch as _mod_branch
from .. import conflicts as _mod_conflicts
from .. import controldir as _mod_controldir
from .. import errors, globbing, lock, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, urlutils, workingtree
from ..decorators import only_raises
from ..mutabletree import BadReferenceTarget, MutableTree
from .dir import BareLocalGitControlDirFormat, LocalGitDir
from .mapping import decode_git_path, encode_git_path, mode_kind
from .tree import MutableGitIndexTree
def _lock_write_tree(self):
    if not self._lock_mode:
        self._lock_mode = 'w'
        self._lock_count = 1
        try:
            self._index_file = GitFile(self.control_transport.local_abspath('index'), 'wb')
        except FileLocked:
            raise errors.LockContention('index')
        self._read_index()
    elif self._lock_mode == 'r':
        raise errors.ReadOnlyError(self)
    else:
        self._lock_count += 1