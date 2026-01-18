import urllib.parse as urlparse
from glance.i18n import _
class MaxRedirectsExceeded(GlanceException):
    message = _('Maximum redirects (%(redirects)s) was exceeded.')