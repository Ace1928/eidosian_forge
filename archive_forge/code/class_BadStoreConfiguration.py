import urllib.parse
from glance_store.i18n import _
class BadStoreConfiguration(GlanceStoreException):
    message = _('Store %(store_name)s could not be configured correctly. Reason: %(reason)s')