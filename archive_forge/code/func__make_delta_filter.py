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
def _make_delta_filter(branch, generate_delta, search, log_rev_iterator, files=None, direction='reverse'):
    """Add revision deltas to a log iterator if needed.

    :param branch: The branch being logged.
    :param generate_delta: Whether to generate a delta for each revision.
      Permitted values are None, 'full' and 'partial'.
    :param search: A user text search string.
    :param log_rev_iterator: An input iterator containing all revisions that
        could be displayed, in lists.
    :param files: If non empty, only revisions matching one or more of
      the files are to be kept.
    :param direction: the direction in which view_revisions is sorted
    :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
        delta).
    """
    if not generate_delta and (not files):
        return log_rev_iterator
    return _generate_deltas(branch.repository, log_rev_iterator, generate_delta, files, direction)