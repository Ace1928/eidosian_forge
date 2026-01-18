import errno
class StreamMismatch(ZFSError):
    errno = errno.ENODEV
    message = 'Stream is not applicable to destination dataset'

    def __init__(self, name):
        self.name = name