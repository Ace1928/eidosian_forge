import errno
class BookmarkNotFound(ZFSError):
    errno = errno.ENOENT
    message = 'Bookmark not found'

    def __init__(self, name):
        self.name = name