from breezy.errors import BzrError, DependencyNotPresent
from breezy.branch import Branch
class Flake8Errors(BzrError):
    _fmt = 'Running in strict flake8 mode; aborting commit, since %(errors)d flake8 errors exist.'

    def __init__(self, errors):
        self.errors = errors