import errno
class QuotaExceeded(ZFSError):
    errno = errno.EDQUOT
    message = 'Quouta exceeded'

    def __init__(self, name):
        self.name = name