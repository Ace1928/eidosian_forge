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
class InventoryTreeChange(TreeChange):
    __slots__ = TreeChange.__slots__ + ['file_id', 'parent_id']

    def __init__(self, file_id, path, changed_content, versioned, parent_id, name, kind, executable, copied=False):
        self.file_id = file_id
        self.parent_id = parent_id
        super().__init__(path=path, changed_content=changed_content, versioned=versioned, name=name, kind=kind, executable=executable, copied=copied)

    def __repr__(self):
        return '{}{!r}'.format(self.__class__.__name__, self._as_tuple())

    def _as_tuple(self):
        return (self.file_id, self.path, self.changed_content, self.versioned, self.parent_id, self.name, self.kind, self.executable, self.copied)

    def __eq__(self, other):
        if isinstance(other, TreeChange):
            return self._as_tuple() == other._as_tuple()
        if isinstance(other, tuple):
            return self._as_tuple() == other
        return False

    def __lt__(self, other):
        return self._as_tuple() < other._as_tuple()

    def meta_modified(self):
        if self.versioned == (True, True):
            return self.executable[0] != self.executable[1]
        return False

    def is_reparented(self):
        return self.parent_id[0] != self.parent_id[1]

    @property
    def renamed(self):
        return not self.copied and None not in self.name and (None not in self.parent_id) and (self.name[0] != self.name[1] or self.parent_id[0] != self.parent_id[1])

    def discard_new(self):
        return self.__class__(self.file_id, (self.path[0], None), self.changed_content, (self.versioned[0], None), (self.parent_id[0], None), (self.name[0], None), (self.kind[0], None), (self.executable[0], None), copied=False)