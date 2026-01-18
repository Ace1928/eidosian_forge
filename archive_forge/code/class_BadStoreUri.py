import urllib.parse as urlparse
from glance.i18n import _
class BadStoreUri(GlanceException):
    message = _('The Store URI was malformed.')