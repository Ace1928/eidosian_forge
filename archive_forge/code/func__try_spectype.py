from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _try_spectype(self, rstype, branch):
    rs = rstype(self.spec, _internal=True)
    return rs.in_history(branch)