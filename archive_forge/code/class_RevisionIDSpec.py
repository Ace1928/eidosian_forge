from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionIDSpec(RevisionSpec):

    def _match_on(self, branch, revs):
        revision_id = self.as_revision_id(branch)
        return RevisionInfo.from_revision_id(branch, revision_id)