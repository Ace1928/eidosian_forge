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
def _parent_loops(self):
    """No entry should be its own ancestor"""
    for trans_id in self._new_parent:
        seen = set()
        parent_id = trans_id
        while parent_id != ROOT_PARENT:
            seen.add(parent_id)
            try:
                parent_id = self.final_parent(parent_id)
            except KeyError:
                break
            if parent_id == trans_id:
                yield ('parent loop', trans_id)
            if parent_id in seen:
                break