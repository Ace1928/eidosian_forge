import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
class LogRevision:
    """A revision to be logged (by LogFormatter.log_revision).

    A simple wrapper for the attributes of a revision to be logged.
    The attributes may or may not be populated, as determined by the
    logging options and the log formatter capabilities.
    """

    def __init__(self, rev=None, revno=None, merge_depth=0, delta=None, tags=None, diff=None, signature=None):
        self.rev = rev
        if revno is None:
            self.revno = None
        else:
            self.revno = str(revno)
        self.merge_depth = merge_depth
        self.delta = delta
        self.tags = tags
        self.diff = diff
        self.signature = signature