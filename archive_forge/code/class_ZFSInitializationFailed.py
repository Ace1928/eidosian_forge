import errno
class ZFSInitializationFailed(ZFSError):
    message = 'Failed to initialize libzfs_core'

    def __init__(self, errno):
        self.errno = errno