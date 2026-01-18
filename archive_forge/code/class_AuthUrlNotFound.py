import urllib.parse as urlparse
from glance.i18n import _
class AuthUrlNotFound(GlanceException):
    message = _('Auth service at URL %(url)s not found.')