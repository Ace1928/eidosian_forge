from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_last(RevisionSpec):
    """Selects the nth revision from the end."""
    help_txt = "Selects the nth revision from the end.\n\n    Supply a positive number to get the nth revision from the end.\n    This is the same as supplying negative numbers to the 'revno:' spec.\n    Examples::\n\n      last:1        -> return the last revision\n      last:3        -> return the revision 2 before the end.\n    "
    prefix = 'last:'

    def _match_on(self, branch, revs):
        revno, revision_id = self._revno_and_revision_id(branch)
        return RevisionInfo(branch, revno, revision_id)

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

    def _as_revision_id(self, context_branch):
        revno, revision_id = self._revno_and_revision_id(context_branch)
        return revision_id