import urllib.parse as urlparse
from glance.i18n import _
class ReservedProperty(Forbidden):
    message = _("Attribute '%(property)s' is reserved.")