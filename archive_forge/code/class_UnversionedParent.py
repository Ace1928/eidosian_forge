import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class UnversionedParent(HandledConflict):
    """An attempt to version a file whose parent directory is not versioned.
    Typically, the result of a merge where one tree unversioned the directory
    and the other added a versioned file to it.
    """
    typestring = 'unversioned parent'
    format = 'Conflict because %(path)s is not versioned, but has versioned children.  %(action)s.'

    def action_take_this(self, tree):
        pass

    def action_take_other(self, tree):
        pass