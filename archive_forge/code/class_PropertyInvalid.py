import errno
class PropertyInvalid(ZFSError):
    errno = errno.EINVAL
    message = 'Invalid property or property value'

    def __init__(self, name):
        self.name = name