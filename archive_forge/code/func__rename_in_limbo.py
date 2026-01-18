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
def _rename_in_limbo(self, trans_ids):
    """Fix limbo names so that the right final path is produced.

        This means we outsmarted ourselves-- we tried to avoid renaming
        these files later by creating them with their final names in their
        final parents.  But now the previous name or parent is no longer
        suitable, so we have to rename them.

        Even for trans_ids that have no new contents, we must remove their
        entries from _limbo_files, because they are now stale.
        """
    for trans_id in trans_ids:
        old_path = self._limbo_files[trans_id]
        self._possibly_stale_limbo_files.add(old_path)
        del self._limbo_files[trans_id]
        if trans_id not in self._new_contents:
            continue
        new_path = self._limbo_name(trans_id)
        os.rename(old_path, new_path)
        self._possibly_stale_limbo_files.remove(old_path)
        for descendant in self._limbo_descendants(trans_id):
            desc_path = self._limbo_files[descendant]
            desc_path = new_path + desc_path[len(old_path):]
            self._limbo_files[descendant] = desc_path