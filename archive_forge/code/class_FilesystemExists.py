import errno
class FilesystemExists(DatasetExists):
    message = 'Filesystem already exists'

    def __init__(self, name):
        self.name = name