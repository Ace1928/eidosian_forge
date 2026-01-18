import urllib.parse as urlparse
from glance.i18n import _
class InvalidTaskType(TaskException, Invalid):
    message = _('Provided type of task is unsupported: %(type)s')