import errno
class SnapshotIsCloned(ZFSError):
    errno = errno.EEXIST
    message = 'Snapshot is cloned'

    def __init__(self, name):
        self.name = name