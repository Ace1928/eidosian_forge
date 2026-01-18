import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def _graph_view_revisions(branch, start_rev_id, end_rev_id, rebase_initial_depths=True):
    """Calculate revisions to view including merges, newest to oldest.

    :param branch: the branch
    :param start_rev_id: the lower revision-id
    :param end_rev_id: the upper revision-id
    :param rebase_initial_depth: should depths be rebased until a mainline
      revision is found?
    :return: An iterator of (revision_id, dotted_revno, merge_depth) tuples.
    """
    view_revisions = branch.iter_merge_sorted_revisions(start_revision_id=end_rev_id, stop_revision_id=start_rev_id, stop_rule='with-merges')
    if not rebase_initial_depths:
        for rev_id, merge_depth, revno, end_of_merge in view_revisions:
            yield (rev_id, '.'.join(map(str, revno)), merge_depth)
    else:
        depth_adjustment = None
        for rev_id, merge_depth, revno, end_of_merge in view_revisions:
            if depth_adjustment is None:
                depth_adjustment = merge_depth
            if depth_adjustment:
                if merge_depth < depth_adjustment:
                    depth_adjustment = merge_depth
                merge_depth -= depth_adjustment
            yield (rev_id, '.'.join(map(str, revno)), merge_depth)