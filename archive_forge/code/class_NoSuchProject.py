import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class NoSuchProject(errors.BzrError):
    _fmt = 'Project does not exist: %(project)s.'

    def __init__(self, project):
        errors.BzrError.__init__(self)
        self.project = project