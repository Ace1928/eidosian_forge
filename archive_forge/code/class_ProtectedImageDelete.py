import urllib.parse as urlparse
from glance.i18n import _
class ProtectedImageDelete(Forbidden):
    message = _('Image %(image_id)s is protected and cannot be deleted.')