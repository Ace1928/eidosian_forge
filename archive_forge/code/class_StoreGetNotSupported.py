import urllib.parse
from glance_store.i18n import _
class StoreGetNotSupported(GlanceStoreException):
    message = _('Getting images from this store is not supported.')