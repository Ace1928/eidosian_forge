import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class UnsupportedForge(errors.BzrError):
    _fmt = 'No supported forge for %(branch)s.'

    def __init__(self, branch):
        errors.BzrError.__init__(self)
        self.branch = branch