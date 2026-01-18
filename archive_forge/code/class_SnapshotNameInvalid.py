import errno
class SnapshotNameInvalid(NameInvalid):
    message = 'Invalid name for snapshot'

    def __init__(self, name):
        self.name = name