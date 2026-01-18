import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class DuplicateEntry(HandledPathConflict):
    """Two directory entries want to have the same name."""
    typestring = 'duplicate'
    format = 'Conflict adding file %(conflict_path)s.  %(action)s %(path)s.'

    def action_take_this(self, tree):
        tree.remove([self.conflict_path], force=True, keep_files=False)
        tree.rename_one(self.path, self.conflict_path)

    def action_take_other(self, tree):
        tree.remove([self.path], force=True, keep_files=False)