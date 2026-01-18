import urllib.parse as urlparse
from glance.i18n import _
class AuthorizationRedirect(GlanceException):
    message = _('Redirecting to %(uri)s for authorization.')