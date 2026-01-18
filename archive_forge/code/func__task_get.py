import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def _task_get(context, task_id, force_show_deleted=False):
    try:
        task = DATA['tasks'][task_id]
    except KeyError:
        msg = _LW('Could not find task %s') % task_id
        LOG.warning(msg)
        raise exception.TaskNotFound(task_id=task_id)
    if task['deleted'] and (not (force_show_deleted or context.can_see_deleted)):
        msg = _LW('Unable to get deleted task %s') % task_id
        LOG.warning(msg)
        raise exception.TaskNotFound(task_id=task_id)
    if not _is_task_visible(context, task):
        LOG.debug('Forbidding request, task %s is not visible', task_id)
        msg = _('Forbidding request, task %s is not visible') % task_id
        raise exception.Forbidden(msg)
    task_info = _task_info_get(task_id)
    return (task, task_info)