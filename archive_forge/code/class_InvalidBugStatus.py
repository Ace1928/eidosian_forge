from . import errors, registry, urlutils
class InvalidBugStatus(errors.BzrError):
    _fmt = "Invalid bug status: '%(status)s'"

    def __init__(self, status):
        self.status = status