from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
class UpgradeChangesContent(BzrError):
    """Inconsistency was found upgrading the mapping of a revision."""
    _fmt = 'Upgrade will change contents in revision %(revid)s. Use --allow-changes to override.'

    def __init__(self, revid):
        self.revid = revid