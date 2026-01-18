from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _as_tree(self, context_branch):
    from .branch import Branch
    other_branch = Branch.open(self.spec)
    last_revision = other_branch.last_revision()
    if last_revision == revision.NULL_REVISION:
        raise errors.NoCommits(other_branch)
    return other_branch.repository.revision_tree(last_revision)