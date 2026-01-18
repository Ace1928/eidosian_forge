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
def action_auto(self, tree):
    try:
        kind = tree.kind(self.path)
    except _mod_transport.NoSuchFile:
        return
    if kind != 'file':
        raise NotImplementedError('Conflict is not a file')
    conflict_markers_in_line = self._conflict_re.search
    with tree.get_file(self.path) as f:
        for line in f:
            if conflict_markers_in_line(line):
                raise NotImplementedError('Conflict markers present')