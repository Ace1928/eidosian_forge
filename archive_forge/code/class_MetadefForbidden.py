import urllib.parse as urlparse
from glance.i18n import _
class MetadefForbidden(Forbidden):
    message = _('You are not authorized to complete this action.')