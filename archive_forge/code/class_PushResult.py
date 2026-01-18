from . import branch as _mod_branch
from . import controldir, errors
from . import revision as _mod_revision
from . import transport
from .i18n import gettext
from .trace import note, warning
class PushResult:
    """Result of a push operation.

    :ivar branch_push_result: Result of a push between branches
    :ivar target_branch: The target branch
    :ivar stacked_on: URL of the branch on which the result is stacked
    :ivar workingtree_updated: Whether or not the target workingtree was updated.
    """

    def __init__(self):
        self.branch_push_result = None
        self.stacked_on = None
        self.workingtree_updated = None
        self.target_branch = None

    def report(self, to_file):
        """Write a human-readable description of the result."""
        if self.branch_push_result is None:
            if self.stacked_on is not None:
                note(gettext('Created new stacked branch referring to %s.') % self.stacked_on)
            else:
                note(gettext('Created new branch.'))
        else:
            self.branch_push_result.report(to_file)