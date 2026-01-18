from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_revid(RevisionIDSpec):
    """Selects a revision using the revision id."""
    help_txt = "Selects a revision using the revision id.\n\n    Supply a specific revision id, that can be used to specify any\n    revision id in the ancestry of the branch.\n    Including merges, and pending merges.\n    Examples::\n\n      revid:aaaa@bbbb-123456789 -> Select revision 'aaaa@bbbb-123456789'\n    "
    prefix = 'revid:'

    def _as_revision_id(self, context_branch):
        if isinstance(self.spec, str):
            return self.spec.encode('utf-8')
        return self.spec