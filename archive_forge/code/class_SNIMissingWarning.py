from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class SNIMissingWarning(HTTPWarning):
    """Warned when making a HTTPS request without SNI available."""
    pass