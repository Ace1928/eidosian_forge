import errno
class FeatureNotSupported(ZFSError):
    errno = errno.ENOTSUP
    message = 'Feature is not supported in this version'

    def __init__(self, name):
        self.name = name