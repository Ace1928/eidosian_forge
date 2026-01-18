import urllib.parse as urlparse
from glance.i18n import _
class ImageMemberLimitExceeded(LimitExceeded):
    message = _('The limit has been exceeded on the number of allowed image members for this image. Attempted: %(attempted)s, Maximum: %(maximum)s')