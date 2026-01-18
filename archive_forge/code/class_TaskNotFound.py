import urllib.parse as urlparse
from glance.i18n import _
class TaskNotFound(TaskException, NotFound):
    message = _('Task with the given id %(task_id)s was not found')