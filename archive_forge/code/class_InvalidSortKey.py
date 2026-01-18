import urllib.parse as urlparse
from glance.i18n import _
class InvalidSortKey(Invalid):
    message = _('Sort key supplied was not valid.')