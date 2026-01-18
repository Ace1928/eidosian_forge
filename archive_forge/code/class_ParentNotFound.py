import errno
class ParentNotFound(ZFSError):
    errno = errno.ENOENT
    message = 'Parent not found'

    def __init__(self, name):
        self.name = name