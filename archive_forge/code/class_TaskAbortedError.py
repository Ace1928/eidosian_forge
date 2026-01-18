import urllib.parse as urlparse
from glance.i18n import _
class TaskAbortedError(ImportTaskError):
    message = _('Task was aborted externally')