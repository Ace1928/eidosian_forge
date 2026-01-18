import urllib.parse
from glance_store.i18n import _
class StoreDeleteNotSupported(GlanceStoreException):
    message = _('Deleting images from this store is not supported.')