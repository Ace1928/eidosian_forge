import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class LabelsUnsupported(errors.BzrError):
    """Labels not supported by this forge."""
    _fmt = 'Labels are not supported by %(forge)r.'

    def __init__(self, forge):
        errors.BzrError.__init__(self)
        self.forge = forge