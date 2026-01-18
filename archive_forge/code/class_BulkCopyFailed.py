from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class BulkCopyFailed(FSError):
    """A copy operation failed in worker threads."""
    default_message = 'One or more copy operations failed (see errors attribute)'

    def __init__(self, errors):
        self.errors = errors
        super(BulkCopyFailed, self).__init__()