import errno
class MultipleOperationsFailure(ZFSError):

    def __init__(self, errors, suppressed_count):
        self.errno = errors[0].errno
        self.errors = errors
        self.suppressed_count = suppressed_count

    def __str__(self):
        return '%s, %d errors included, %d suppressed' % (ZFSError.__str__(self), len(self.errors), self.suppressed_count)

    def __repr__(self):
        return '%s(%r, %r, errors=%r, supressed=%r)' % (self.__class__.__name__, self.errno, self.message, self.errors, self.suppressed_count)