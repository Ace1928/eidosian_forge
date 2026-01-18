import urllib.parse as urlparse
from glance.i18n import _
class ImportTaskError(TaskException, Invalid):
    message = _('An import task exception occurred')