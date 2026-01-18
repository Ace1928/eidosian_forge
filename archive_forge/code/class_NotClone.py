import errno
class NotClone(ZFSError):
    errno = errno.EINVAL
    message = 'Filesystem is not a clone, can not promote'

    def __init__(self, name):
        self.name = name