import urllib.parse as urlparse
from glance.i18n import _
class ReadonlyProperty(Forbidden):
    message = _("Attribute '%(property)s' is read-only.")