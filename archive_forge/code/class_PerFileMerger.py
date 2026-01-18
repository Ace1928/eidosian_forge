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
class PerFileMerger(AbstractPerFileMerger):
    """Merge individual files when self.file_matches returns True.

    This class is intended to be subclassed.  The file_matches and
    merge_matching methods should be overridden with concrete implementations.
    """

    def file_matches(self, params):
        """Return True if merge_matching should be called on this file.

        Only called with merges of plain files with no clear winner.

        Subclasses must override this.
        """
        raise NotImplementedError(self.file_matches)

    def merge_contents(self, params):
        """Merge the contents of a single file."""
        if params.winner == 'other' or not params.is_file_merge() or (not self.file_matches(params)):
            return ('not_applicable', None)
        return self.merge_matching(params)

    def merge_matching(self, params):
        """Merge the contents of a single file that has matched the criteria
        in PerFileMerger.merge_contents (is a conflict, is a file,
        self.file_matches is True).

        Subclasses must override this.
        """
        raise NotImplementedError(self.merge_matching)