import errno
class DuplicateSnapshots(ZFSError):
    errno = errno.EXDEV
    message = 'Requested multiple snapshots of the same filesystem'

    def __init__(self, name):
        self.name = name