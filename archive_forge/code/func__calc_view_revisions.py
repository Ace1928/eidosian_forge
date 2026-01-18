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
def _calc_view_revisions(branch, start_rev_id, end_rev_id, direction, generate_merge_revisions, delayed_graph_generation=False, exclude_common_ancestry=False):
    """Calculate the revisions to view.

    :return: An iterator of (revision_id, dotted_revno, merge_depth) tuples OR
             a list of the same tuples.
    """
    if exclude_common_ancestry and start_rev_id == end_rev_id:
        raise errors.CommandError(gettext('--exclude-common-ancestry requires two different revisions'))
    if direction not in ('reverse', 'forward'):
        raise ValueError(gettext('invalid direction %r') % direction)
    br_rev_id = branch.last_revision()
    if br_rev_id == _mod_revision.NULL_REVISION:
        return []
    if end_rev_id and start_rev_id == end_rev_id and (not generate_merge_revisions or not _has_merges(branch, end_rev_id)):
        return _generate_one_revision(branch, end_rev_id, br_rev_id, branch.revno())
    if not generate_merge_revisions:
        try:
            iter_revs = _linear_view_revisions(branch, start_rev_id, end_rev_id, exclude_common_ancestry=exclude_common_ancestry)
            if direction == 'forward' or (start_rev_id and (not _is_obvious_ancestor(branch, start_rev_id, end_rev_id))):
                iter_revs = list(iter_revs)
            if direction == 'forward':
                iter_revs = reversed(iter_revs)
            return iter_revs
        except _StartNotLinearAncestor:
            pass
    iter_revs = _generate_all_revisions(branch, start_rev_id, end_rev_id, direction, delayed_graph_generation, exclude_common_ancestry)
    if direction == 'forward':
        iter_revs = _rebase_merge_depth(reverse_by_depth(list(iter_revs)))
    return iter_revs