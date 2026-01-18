import urllib.parse as urlparse
from glance.i18n import _
class ImageSizeLimitExceeded(GlanceException):
    message = _('The provided image is too large.')