import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class PrerequisiteBranchUnsupported(errors.BzrError):
    """Prerequisite branch not supported by this forge."""

    def __init__(self, forge):
        errors.BzrError.__init__(self)
        self.forge = forge