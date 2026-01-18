import py, sys
class DeprecationWarning(DeprecationWarning):

    def __init__(self, msg, path, lineno):
        self.msg = msg
        self.path = path
        self.lineno = lineno

    def __repr__(self):
        return '%s:%d: %s' % (self.path, self.lineno + 1, self.msg)

    def __str__(self):
        return self.msg