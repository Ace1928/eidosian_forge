import urllib.parse as urlparse
from glance.i18n import _
class MissingCredentialError(GlanceException):
    message = _('Missing required credential: %(required)s')