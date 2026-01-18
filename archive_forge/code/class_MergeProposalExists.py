import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class MergeProposalExists(errors.BzrError):
    _fmt = 'A merge proposal already exists: %(url)s.'

    def __init__(self, url, existing_proposal=None):
        errors.BzrError.__init__(self)
        self.url = url
        self.existing_proposal = existing_proposal