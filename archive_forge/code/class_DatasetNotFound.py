import errno
class DatasetNotFound(ZFSError):
    """
    This exception is raised when an operation failure can be caused by a missing
    snapshot or a missing filesystem and it is impossible to distinguish between
    the causes.
    """
    errno = errno.ENOENT
    message = 'Dataset not found'

    def __init__(self, name):
        self.name = name