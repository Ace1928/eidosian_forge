import errno
class NameInvalid(ZFSError):
    errno = errno.EINVAL
    message = 'Invalid name'

    def __init__(self, name):
        self.name = name