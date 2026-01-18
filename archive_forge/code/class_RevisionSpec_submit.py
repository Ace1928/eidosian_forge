from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_submit(RevisionSpec_ancestor):
    """Selects a common ancestor with a submit branch."""
    help_txt = 'Selects a common ancestor with the submit branch.\n\n    Diffing against this shows all the changes that were made in this branch,\n    and is a good predictor of what merge will do.  The submit branch is\n    used by the bundle and merge directive commands.  If no submit branch\n    is specified, the parent branch is used instead.\n\n    The common ancestor is the last revision that existed in both\n    branches. Usually this is the branch point, but it could also be\n    a revision that was merged.\n\n    Examples::\n\n      $ bzr diff -r submit:\n    '
    prefix = 'submit:'

    def _get_submit_location(self, branch):
        submit_location = branch.get_submit_branch()
        location_type = 'submit branch'
        if submit_location is None:
            submit_location = branch.get_parent()
            location_type = 'parent branch'
        if submit_location is None:
            raise errors.NoSubmitBranch(branch)
        trace.note(gettext('Using {0} {1}').format(location_type, submit_location))
        return submit_location

    def _match_on(self, branch, revs):
        trace.mutter('matching ancestor: on: %s, %s', self.spec, branch)
        return self._find_revision_info(branch, self._get_submit_location(branch))

    def _as_revision_id(self, context_branch):
        return self._find_revision_id(context_branch, self._get_submit_location(context_branch))