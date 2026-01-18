from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_branch(RevisionSpec):
    """Selects the last revision of a specified branch."""
    help_txt = 'Selects the last revision of a specified branch.\n\n    Supply the path to a branch to select its last revision.\n\n    Examples::\n\n      branch:/path/to/branch\n    '
    prefix = 'branch:'
    dwim_catchable_exceptions = [errors.NotBranchError]

    def _match_on(self, branch, revs):
        from .branch import Branch
        other_branch = Branch.open(self.spec)
        revision_b = other_branch.last_revision()
        if revision_b in (None, revision.NULL_REVISION):
            raise errors.NoCommits(other_branch)
        if branch is None:
            branch = other_branch
        else:
            try:
                branch.fetch(other_branch, revision_b)
            except errors.ReadOnlyError:
                branch = other_branch
        return RevisionInfo(branch, None, revision_b)

    def _as_revision_id(self, context_branch):
        from .branch import Branch
        other_branch = Branch.open(self.spec)
        last_revision = other_branch.last_revision()
        context_branch.fetch(other_branch, last_revision)
        if last_revision == revision.NULL_REVISION:
            raise errors.NoCommits(other_branch)
        return last_revision

    def _as_tree(self, context_branch):
        from .branch import Branch
        other_branch = Branch.open(self.spec)
        last_revision = other_branch.last_revision()
        if last_revision == revision.NULL_REVISION:
            raise errors.NoCommits(other_branch)
        return other_branch.repository.revision_tree(last_revision)

    def needs_branch(self):
        return False

    def get_branch(self):
        return self.spec