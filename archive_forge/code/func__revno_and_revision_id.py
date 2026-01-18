from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _revno_and_revision_id(self, context_branch):
    last_revno, last_revision_id = context_branch.last_revision_info()
    if self.spec == '':
        if not last_revno:
            raise errors.NoCommits(context_branch)
        return (last_revno, last_revision_id)
    try:
        offset = int(self.spec)
    except ValueError as e:
        raise InvalidRevisionSpec(self.user_spec, context_branch, e)
    if offset <= 0:
        raise InvalidRevisionSpec(self.user_spec, context_branch, 'you must supply a positive value')
    revno = last_revno - offset + 1
    try:
        revision_id = context_branch.get_rev_id(revno)
    except (errors.NoSuchRevision, errors.RevnoOutOfBounds):
        raise InvalidRevisionSpec(self.user_spec, context_branch)
    return (revno, revision_id)