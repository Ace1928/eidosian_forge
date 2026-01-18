import urllib.parse as urlparse
from glance.i18n import _
class InvalidTaskStatusTransition(TaskException, Invalid):
    message = _('Status transition from %(cur_status)s to %(new_status)s is not allowed')