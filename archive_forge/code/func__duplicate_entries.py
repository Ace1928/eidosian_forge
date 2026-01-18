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
def _duplicate_entries(self, by_parent):
    """No directory may have two entries with the same name."""
    if (self._new_name, self._new_parent) == ({}, {}):
        return
    for children in by_parent.values():
        name_ids = []
        for child_tid in children:
            name = self.final_name(child_tid)
            if name is not None:
                if not self._case_sensitive_target:
                    name = name.lower()
                name_ids.append((name, child_tid))
        name_ids.sort()
        last_name = None
        last_trans_id = None
        for name, trans_id in name_ids:
            kind = self.final_kind(trans_id)
            if kind is None and (not self.final_is_versioned(trans_id)):
                continue
            if name == last_name:
                yield ('duplicate', last_trans_id, trans_id, name)
            last_name = name
            last_trans_id = trans_id