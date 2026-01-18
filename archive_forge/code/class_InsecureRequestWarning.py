from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class InsecureRequestWarning(SecurityWarning):
    """Warned when making an unverified HTTPS request."""
    pass