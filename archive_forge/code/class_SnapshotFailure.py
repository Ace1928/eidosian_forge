import errno
class SnapshotFailure(MultipleOperationsFailure):
    message = 'Creation of snapshot(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(SnapshotFailure, self).__init__(errors, suppressed_count)