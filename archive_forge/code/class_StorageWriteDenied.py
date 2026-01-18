import urllib.parse
from glance_store.i18n import _
class StorageWriteDenied(GlanceStoreException):
    message = _('Permission to write image storage media denied.')