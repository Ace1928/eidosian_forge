import urllib.parse as urlparse
from glance.i18n import _
class InvalidTaskStatus(TaskException, Invalid):
    message = _('Provided status of task is unsupported: %(status)s')