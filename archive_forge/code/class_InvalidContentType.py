import urllib.parse as urlparse
from glance.i18n import _
class InvalidContentType(GlanceException):
    message = _('Invalid content type %(content_type)s')