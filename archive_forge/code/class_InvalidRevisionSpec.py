from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class InvalidRevisionSpec(errors.BzrError):
    _fmt = "Requested revision: '%(spec)s' does not exist in branch: %(branch_url)s%(extra)s"

    def __init__(self, spec, branch, extra=None):
        errors.BzrError.__init__(self, branch=branch, spec=spec)
        self.branch_url = getattr(branch, 'user_url', str(branch))
        if extra:
            self.extra = '\n' + str(extra)
        else:
            self.extra = ''