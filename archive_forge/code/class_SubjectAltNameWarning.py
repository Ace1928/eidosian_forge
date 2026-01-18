from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class SubjectAltNameWarning(SecurityWarning):
    """Warned when connecting to a host with a certificate missing a SAN."""
    pass