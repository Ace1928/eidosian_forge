import urllib.parse as urlparse
from glance.i18n import _
class ImageTagLimitExceeded(LimitExceeded):
    message = _('The limit has been exceeded on the number of allowed image tags. Attempted: %(attempted)s, Maximum: %(maximum)s')