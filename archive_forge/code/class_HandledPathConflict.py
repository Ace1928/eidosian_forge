import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class HandledPathConflict(HandledConflict):
    """A provisionally-resolved path problem involving two paths.
    This is intended to be a base class.
    """
    rformat = '%(class)s(%(action)r, %(path)r, %(conflict_path)r, %(file_id)r, %(conflict_file_id)r)'

    def __init__(self, action, path, conflict_path, file_id=None, conflict_file_id=None):
        HandledConflict.__init__(self, action, path, file_id)
        self.conflict_path = conflict_path
        if isinstance(conflict_file_id, str):
            conflict_file_id = cache_utf8.encode(conflict_file_id)
        self.conflict_file_id = conflict_file_id

    def _cmp_list(self):
        return HandledConflict._cmp_list(self) + [self.conflict_path, self.conflict_file_id]

    def as_stanza(self):
        s = HandledConflict.as_stanza(self)
        s.add('conflict_path', self.conflict_path)
        if self.conflict_file_id is not None:
            s.add('conflict_file_id', self.conflict_file_id.decode('utf8'))
        return s