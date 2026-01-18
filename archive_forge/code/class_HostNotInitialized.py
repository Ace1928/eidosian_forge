import urllib.parse
from glance_store.i18n import _
class HostNotInitialized(GlanceStoreException):
    message = _("The glance cinder store host %(host)s which will used to perform nfs mount/umount operations isn't initialized.")