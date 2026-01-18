import errno
class FilesystemNameInvalid(NameInvalid):
    message = 'Invalid name for filesystem or volume'

    def __init__(self, name):
        self.name = name