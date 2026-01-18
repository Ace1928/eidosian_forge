import urllib.parse as urlparse
from glance.i18n import _
class NotAuthenticated(GlanceException):
    message = _('You are not authenticated.')