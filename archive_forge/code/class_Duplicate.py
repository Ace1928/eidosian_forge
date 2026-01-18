import urllib.parse as urlparse
from glance.i18n import _
class Duplicate(GlanceException):
    message = _('An object with the same identifier already exists.')