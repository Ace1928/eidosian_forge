from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _scan_backwards(self, branch, dt):
    with branch.lock_read():
        graph = branch.repository.get_graph()
        last_match = None
        for revid in graph.iter_lefthand_ancestry(branch.last_revision(), (_mod_revision.NULL_REVISION,)):
            r = branch.repository.get_revision(revid)
            if r.datetime() < dt:
                if last_match is None:
                    raise InvalidRevisionSpec(self.user_spec, branch)
                return RevisionInfo(branch, None, last_match)
            last_match = revid
        return RevisionInfo(branch, None, last_match)