from . import errors, registry, urlutils
class InvalidLineInBugsProperty(errors.BzrError):
    _fmt = "Invalid line in bugs property: '%(line)s'"

    def __init__(self, line):
        self.line = line