import errno
class HoldNotFound(ZFSError):
    errno = errno.ENOENT
    message = 'Hold with a given tag does not exist on snapshot'

    def __init__(self, name):
        self.name = name