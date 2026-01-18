import urllib.parse as urlparse
from glance.i18n import _
class ImagePropertyLimitExceeded(LimitExceeded):
    message = _('The limit has been exceeded on the number of allowed image properties. Attempted: %(attempted)s, Maximum: %(maximum)s')