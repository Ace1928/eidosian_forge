import errno
class ZFSError(Exception):
    errno = None
    message = None
    name = None

    def __str__(self):
        if self.name is not None:
            return "[Errno %d] %s: '%s'" % (self.errno, self.message, self.name)
        else:
            return '[Errno %d] %s' % (self.errno, self.message)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.errno, self.message)