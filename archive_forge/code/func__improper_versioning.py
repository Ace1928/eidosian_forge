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
def _improper_versioning(self):
    """Cannot version a file with no contents, or a bad type.

        However, existing entries with no contents are okay.
        """
    for trans_id in self._versioned:
        kind = self.final_kind(trans_id)
        if kind == 'symlink' and (not self._tree.supports_symlinks()):
            continue
        if kind is None:
            yield ('versioning no contents', trans_id)
            continue
        if not self._tree.versionable_kind(kind):
            yield ('versioning bad kind', trans_id, kind)