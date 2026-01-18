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
def _has_named_child(self, name, parent_id, known_children):
    """Does a parent already have a name child.

        :param name: The searched for name.

        :param parent_id: The parent for which the check is made.

        :param known_children: The already known children. This should have
            been recently obtained from `self.by_parent.get(parent_id)`
            (or will be if None is passed).
        """
    if known_children is None:
        known_children = self.by_parent().get(parent_id, [])
    for child in known_children:
        if self.final_name(child) == name:
            return True
    parent_path = self._tree_id_paths.get(parent_id, None)
    if parent_path is None:
        return False
    child_path = joinpath(parent_path, name)
    child_id = self._tree_path_ids.get(child_path, None)
    if child_id is None:
        return osutils.lexists(self._tree.abspath(child_path))
    else:
        raise AssertionError('child_id is missing: %s, %s, %s' % (name, parent_id, child_id))