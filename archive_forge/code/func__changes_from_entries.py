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
def _changes_from_entries(self, source_entry, target_entry, source_path, target_path):
    """Generate a iter_changes tuple between source_entry and target_entry.

        :param source_entry: An inventory entry from self.source, or None.
        :param target_entry: An inventory entry from self.target, or None.
        :param source_path: The path of source_entry.
        :param target_path: The path of target_entry.
        :return: A tuple, item 0 of which is an iter_changes result tuple, and
            item 1 is True if there are any changes in the result tuple.
        """
    if source_entry is None:
        if target_entry is None:
            return None
        file_id = target_entry.file_id
    else:
        file_id = source_entry.file_id
    if source_entry is not None:
        source_versioned = True
        source_name = source_entry.name
        source_parent = source_entry.parent_id
        source_kind, source_executable, source_stat = self.source._comparison_data(source_entry, source_path)
    else:
        source_versioned = False
        source_name = None
        source_parent = None
        source_kind = None
        source_executable = None
    if target_entry is not None:
        target_versioned = True
        target_name = target_entry.name
        target_parent = target_entry.parent_id
        target_kind, target_executable, target_stat = self.target._comparison_data(target_entry, target_path)
    else:
        target_versioned = False
        target_name = None
        target_parent = None
        target_kind = None
        target_executable = None
    versioned = (source_versioned, target_versioned)
    kind = (source_kind, target_kind)
    changed_content = False
    if source_kind != target_kind:
        changed_content = True
    elif source_kind == 'file':
        if not self.file_content_matches(source_path, target_path, source_stat, target_stat):
            changed_content = True
    elif source_kind == 'symlink':
        if self.source.get_symlink_target(source_path) != self.target.get_symlink_target(target_path):
            changed_content = True
    elif source_kind == 'tree-reference':
        if self.source.get_reference_revision(source_path) != self.target.get_reference_revision(target_path):
            changed_content = True
    parent = (source_parent, target_parent)
    name = (source_name, target_name)
    executable = (source_executable, target_executable)
    if changed_content is not False or versioned[0] != versioned[1] or parent[0] != parent[1] or (name[0] != name[1]) or (executable[0] != executable[1]):
        changes = True
    else:
        changes = False
    return (InventoryTreeChange(file_id, (source_path, target_path), changed_content, versioned, parent, name, kind, executable), changes)