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
def _limbo_name(self, trans_id):
    """Generate the limbo name of a file"""
    limbo_name = self._limbo_files.get(trans_id)
    if limbo_name is None:
        limbo_name = self._generate_limbo_path(trans_id)
        self._limbo_files[trans_id] = limbo_name
    return limbo_name