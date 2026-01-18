from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import units
from taskflow import task
from glance.common import exception as glance_exception
from glance.i18n import _LW
class OptionalTask(task.Task):

    def __init__(self, *args, **kwargs):
        super(OptionalTask, self).__init__(*args, **kwargs)
        self.execute = self._catch_all(self.execute)

    def _catch_all(self, func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                msg = _LW('An optional task has failed, the failure was: %s') % encodeutils.exception_to_unicode(exc)
                LOG.warning(msg)
        return wrapper