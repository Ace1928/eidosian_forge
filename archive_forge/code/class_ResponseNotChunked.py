from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class ResponseNotChunked(ProtocolError, ValueError):
    """Response needs to be chunked in order to read it as chunks."""
    pass