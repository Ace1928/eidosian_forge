import urllib.parse as urlparse
from glance.i18n import _
class ImageLocationLimitExceeded(LimitExceeded):
    message = _('The limit has been exceeded on the number of allowed image locations. Attempted: %(attempted)s, Maximum: %(maximum)s')