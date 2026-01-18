from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class CacheVersionMismatch(Error):
    """Cache version mismatch."""

    def __init__(self, message, actual, requested):
        super(CacheVersionMismatch, self).__init__(message)
        self.actual = actual
        self.requested = requested