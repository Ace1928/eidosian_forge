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
def _make_batch_filter(branch, generate_delta, search, log_rev_iterator):
    """Group up a single large batch into smaller ones.

    :param branch: The branch being logged.
    :param generate_delta: Whether to generate a delta for each revision.
    :param search: A user text search string.
    :param log_rev_iterator: An input iterator containing all revisions that
        could be displayed, in lists.
    :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
        delta).
    """
    num = 9
    for batch in log_rev_iterator:
        batch = iter(batch)
        while True:
            step = [detail for _, detail in zip(range(num), batch)]
            if len(step) == 0:
                break
            yield step
            num = min(int(num * 1.5), 200)