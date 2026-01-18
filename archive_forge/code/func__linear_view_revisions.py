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
def _linear_view_revisions(branch, start_rev_id, end_rev_id):
    repo = branch.repository
    graph = repo.get_graph()
    for revision_id in graph.iter_lefthand_ancestry(end_rev_id, (_mod_revision.NULL_REVISION,)):
        revno = branch.revision_id_to_dotted_revno(revision_id)
        revno_str = '.'.join((str(n) for n in revno))
        if revision_id == start_rev_id:
            yield (revision_id, revno_str, 0)
            break
        yield (revision_id, revno_str, 0)