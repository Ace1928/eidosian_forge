import errno
class BookmarkNameInvalid(NameInvalid):
    message = 'Invalid name for bookmark'

    def __init__(self, name):
        self.name = name