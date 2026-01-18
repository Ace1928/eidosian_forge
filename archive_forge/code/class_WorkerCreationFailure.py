import urllib.parse as urlparse
from glance.i18n import _
class WorkerCreationFailure(GlanceException):
    message = _('Server worker creation failed: %(reason)s.')