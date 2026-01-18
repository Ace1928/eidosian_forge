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
def _make_search_filter(branch, generate_delta, match, log_rev_iterator):
    """Create a filtered iterator of log_rev_iterator matching on a regex.

    :param branch: The branch being logged.
    :param generate_delta: Whether to generate a delta for each revision.
    :param match: A dictionary with properties as keys and lists of strings
        as values. To match, a revision may match any of the supplied strings
        within a single property but must match at least one string for each
        property.
    :param log_rev_iterator: An input iterator containing all revisions that
        could be displayed, in lists.
    :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
        delta).
    """
    if not match:
        return log_rev_iterator
    searchRE = [(k, [lazy_regex.lazy_compile(x, re.IGNORECASE) for x in v]) for k, v in match.items()]
    return _filter_re(searchRE, log_rev_iterator)