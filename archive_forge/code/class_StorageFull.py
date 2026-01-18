import urllib.parse
from glance_store.i18n import _
class StorageFull(GlanceStoreException):
    message = _('There is not enough disk space on the image storage media.')