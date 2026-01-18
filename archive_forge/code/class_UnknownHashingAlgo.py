import urllib.parse
from glance_store.i18n import _
class UnknownHashingAlgo(GlanceStoreException):
    message = _('Unknown hashing algorithm identifier: %(algo)s')