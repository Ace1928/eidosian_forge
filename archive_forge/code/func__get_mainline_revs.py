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
def _get_mainline_revs(branch, start_revision, end_revision):
    """Get the mainline revisions from the branch.

    Generates the list of mainline revisions for the branch.

    :param  branch: The branch containing the revisions.

    :param  start_revision: The first revision to be logged.
            For backwards compatibility this may be a mainline integer revno,
            but for merge revision support a RevisionInfo is expected.

    :param  end_revision: The last revision to be logged.
            For backwards compatibility this may be a mainline integer revno,
            but for merge revision support a RevisionInfo is expected.

    :return: A (mainline_revs, rev_nos, start_rev_id, end_rev_id) tuple.
    """
    branch_revno, branch_last_revision = branch.last_revision_info()
    if branch_revno == 0:
        return (None, None, None, None)
    start_rev_id = None
    if start_revision is None:
        start_revno = 1
    elif isinstance(start_revision, revisionspec.RevisionInfo):
        start_rev_id = start_revision.rev_id
        start_revno = start_revision.revno or 1
    else:
        branch.check_real_revno(start_revision)
        start_revno = start_revision
    end_rev_id = None
    if end_revision is None:
        end_revno = branch_revno
    elif isinstance(end_revision, revisionspec.RevisionInfo):
        end_rev_id = end_revision.rev_id
        end_revno = end_revision.revno or branch_revno
    else:
        branch.check_real_revno(end_revision)
        end_revno = end_revision
    if start_rev_id == _mod_revision.NULL_REVISION or end_rev_id == _mod_revision.NULL_REVISION:
        raise errors.CommandError(gettext('Logging revision 0 is invalid.'))
    if start_revno > end_revno:
        raise errors.CommandError(gettext('Start revision must be older than the end revision.'))
    if end_revno < start_revno:
        return (None, None, None, None)
    cur_revno = branch_revno
    rev_nos = {}
    mainline_revs = []
    graph = branch.repository.get_graph()
    for revision_id in graph.iter_lefthand_ancestry(branch_last_revision, (_mod_revision.NULL_REVISION,)):
        if cur_revno < start_revno:
            rev_nos[revision_id] = cur_revno
            mainline_revs.append(revision_id)
            break
        if cur_revno <= end_revno:
            rev_nos[revision_id] = cur_revno
            mainline_revs.append(revision_id)
        cur_revno -= 1
    else:
        mainline_revs.append(None)
    mainline_revs.reverse()
    return (mainline_revs, rev_nos, start_rev_id, end_rev_id)