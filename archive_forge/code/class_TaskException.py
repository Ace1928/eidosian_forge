import urllib.parse as urlparse
from glance.i18n import _
class TaskException(GlanceException):
    message = _('An unknown task exception occurred')