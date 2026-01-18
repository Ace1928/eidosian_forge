import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
def _add_one_and_parent(self, parent_ie, path, kind, inv_path):
    """Add a new entry to the inventory and automatically add unversioned parents.

        :param parent_ie: Parent inventory entry if known, or None.  If
            None, the parent is looked up by name and used if present, otherwise it
            is recursively added.
        :param path: Filesystem path to add
        :param kind: Kind of new entry (file, directory, etc)
        :param inv_path: Inventory path
        :return: Inventory entry for path and a list of paths which have been added.
        """
    inv_dirname = osutils.dirname(inv_path)
    dirname, basename = osutils.split(path)
    if parent_ie is None:
        this_ie = self._get_ie(inv_path)
        if this_ie is not None:
            return this_ie
        parent_ie = self._add_one_and_parent(None, dirname, 'directory', inv_dirname)
    if parent_ie.kind != 'directory':
        parent_ie = self._convert_to_directory(parent_ie, inv_dirname)
    file_id = self.action(self.tree, parent_ie, path, kind)
    entry = _mod_inventory.make_entry(kind, basename, parent_ie.file_id, file_id=file_id)
    self._invdelta[inv_path] = (None, inv_path, entry.file_id, entry)
    self.added.append(inv_path)
    return entry