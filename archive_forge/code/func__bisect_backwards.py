from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _bisect_backwards(self, branch, dt, hi):
    import bisect
    with branch.lock_read():
        rev = bisect.bisect(_RevListToTimestamps(branch), dt, 1, hi)
    if rev == branch.revno():
        raise InvalidRevisionSpec(self.user_spec, branch)
    return RevisionInfo(branch, rev)