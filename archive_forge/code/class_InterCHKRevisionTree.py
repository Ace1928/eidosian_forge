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
class InterCHKRevisionTree(InterInventoryTree):
    """Fast path optimiser for RevisionTrees with CHK inventories."""

    @staticmethod
    def is_compatible(source, target):
        if isinstance(source, RevisionTree) and isinstance(target, RevisionTree):
            try:
                source.root_inventory.id_to_entry
                target.root_inventory.id_to_entry
                return True
            except AttributeError:
                pass
        return False

    def iter_changes(self, include_unchanged=False, specific_files=None, pb=None, extra_trees=[], require_versioned=True, want_unversioned=False):
        lookup_trees = [self.source]
        if extra_trees:
            lookup_trees.extend(extra_trees)
        precise_file_ids = set()
        discarded_changes = {}
        if specific_files == []:
            specific_file_ids = []
        else:
            specific_file_ids = self.target.paths2ids(specific_files, lookup_trees, require_versioned=require_versioned)
        changed_file_ids = set()
        for result in self.target.root_inventory.iter_changes(self.source.root_inventory):
            result = InventoryTreeChange(*result)
            if specific_file_ids is not None:
                if result.file_id not in specific_file_ids:
                    discarded_changes[result.file_id] = result
                    continue
                precise_file_ids.add(result.parent_id[1])
            yield result
            changed_file_ids.add(result.file_id)
        if specific_file_ids is not None:
            for result in self._handle_precise_ids(precise_file_ids, changed_file_ids, discarded_changes=discarded_changes):
                yield result
        if include_unchanged:
            changed_file_ids = set(changed_file_ids)
            for relpath, entry in self.target.root_inventory.iter_entries():
                if specific_file_ids is not None and entry.file_id not in specific_file_ids:
                    continue
                if entry.file_id not in changed_file_ids:
                    yield InventoryTreeChange(entry.file_id, (relpath, relpath), False, (True, True), (entry.parent_id, entry.parent_id), (entry.name, entry.name), (entry.kind, entry.kind), (entry.executable, entry.executable))