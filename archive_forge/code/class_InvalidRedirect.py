import urllib.parse as urlparse
from glance.i18n import _
class InvalidRedirect(GlanceException):
    message = _('Received invalid HTTP redirect.')