import errno
class BookmarkFailure(MultipleOperationsFailure):
    message = 'Creation of bookmark(s) failed for one or more reasons'

    def __init__(self, errors, suppressed_count):
        super(BookmarkFailure, self).__init__(errors, suppressed_count)