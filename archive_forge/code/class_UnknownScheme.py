import urllib.parse
from glance_store.i18n import _
class UnknownScheme(GlanceStoreException):
    message = _("Unknown scheme '%(scheme)s' found in URI")