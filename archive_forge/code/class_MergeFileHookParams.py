import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class MergeFileHookParams:
    """Object holding parameters passed to merge_file_content hooks.

    There are some fields hooks can access:

    :ivar base_path: Path in base tree
    :ivar other_path: Path in other tree
    :ivar this_path: Path in this tree
    :ivar trans_id: the transform ID for the merge of this file
    :ivar this_kind: kind of file in 'this' tree
    :ivar other_kind: kind of file in 'other' tree
    :ivar winner: one of 'this', 'other', 'conflict'
    """

    def __init__(self, merger, paths, trans_id, this_kind, other_kind, winner):
        self._merger = merger
        self.paths = paths
        self.base_path, self.other_path, self.this_path = paths
        self.trans_id = trans_id
        self.this_kind = this_kind
        self.other_kind = other_kind
        self.winner = winner

    def is_file_merge(self):
        """True if this_kind and other_kind are both 'file'."""
        return self.this_kind == 'file' and self.other_kind == 'file'

    @decorators.cachedproperty
    def base_lines(self):
        """The lines of the 'base' version of the file."""
        return self._merger.get_lines(self._merger.base_tree, self.base_path)

    @decorators.cachedproperty
    def this_lines(self):
        """The lines of the 'this' version of the file."""
        return self._merger.get_lines(self._merger.this_tree, self.this_path)

    @decorators.cachedproperty
    def other_lines(self):
        """The lines of the 'other' version of the file."""
        return self._merger.get_lines(self._merger.other_tree, self.other_path)