from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _set_task_status(self, new_status):
    if self._validate_task_status_transition(self.status, new_status):
        old_status = self.status
        self._status = new_status
        LOG.info(_LI('Task [%(task_id)s] status changing from %(cur_status)s to %(new_status)s'), {'task_id': self.task_id, 'cur_status': old_status, 'new_status': new_status})
    else:
        LOG.error(_LE('Task [%(task_id)s] status failed to change from %(cur_status)s to %(new_status)s'), {'task_id': self.task_id, 'cur_status': self.status, 'new_status': new_status})
        raise exception.InvalidTaskStatusTransition(cur_status=self.status, new_status=new_status)