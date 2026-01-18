import urllib.parse as urlparse
from glance.i18n import _
class InvalidFilterOperatorValue(Invalid):
    message = _('Unable to filter using the specified operator.')