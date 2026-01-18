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
class AbstractPerFileMerger:
    """PerFileMerger objects are used by plugins extending merge for breezy.

    See ``breezy.plugins.news_merge.news_merge`` for an example concrete class.

    :ivar merger: The Merge3Merger performing the merge.
    """

    def __init__(self, merger):
        """Create a PerFileMerger for use with merger."""
        self.merger = merger

    def merge_contents(self, merge_params):
        """Attempt to merge the contents of a single file.

        :param merge_params: A breezy.merge.MergeFileHookParams
        :return: A tuple of (status, chunks), where status is one of
            'not_applicable', 'success', 'conflicted', or 'delete'.  If status
            is 'success' or 'conflicted', then chunks should be an iterable of
            strings for the new file contents.
        """
        return ('not applicable', None)