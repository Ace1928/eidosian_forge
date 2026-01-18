import urllib.parse
from glance_store.i18n import _
class HasSnapshot(GlanceStoreException):
    message = _('The image cannot be deleted because it has snapshot(s).')