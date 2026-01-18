import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class DuplicateID(HandledPathConflict):
    """Two files want the same file_id."""
    typestring = 'duplicate id'
    format = 'Conflict adding id to %(conflict_path)s.  %(action)s %(path)s.'