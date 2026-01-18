import errno
class BookmarkMismatch(ZFSError):
    errno = errno.EINVAL
    message = "Bookmark is not in snapshot's filesystem"

    def __init__(self, name):
        self.name = name