from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
@staticmethod
def _find_revision_info(branch, other_location):
    revision_id = RevisionSpec_ancestor._find_revision_id(branch, other_location)
    return RevisionInfo(branch, None, revision_id)