from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class InvalidChunkLength(HTTPError, httplib_IncompleteRead):
    """Invalid chunk length in a chunked response."""

    def __init__(self, response, length):
        super(InvalidChunkLength, self).__init__(response.tell(), response.length_remaining)
        self.response = response
        self.length = length

    def __repr__(self):
        return 'InvalidChunkLength(got length %r, %i bytes read)' % (self.length, self.partial)