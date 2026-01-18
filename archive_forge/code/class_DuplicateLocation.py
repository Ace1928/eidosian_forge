import urllib.parse as urlparse
from glance.i18n import _
class DuplicateLocation(Duplicate):
    message = _('The location %(location)s already exists')