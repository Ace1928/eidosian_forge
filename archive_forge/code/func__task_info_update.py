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
def _task_info_update(task_id, values):
    """Update Task Info for Task with given task ID and updated values"""
    global DATA
    try:
        task_info = DATA['task_info'][task_id]
    except KeyError:
        LOG.debug('No task info found with task id %s', task_id)
        raise exception.TaskNotFound(task_id=task_id)
    task_info.update(values)
    DATA['task_info'][task_id] = task_info
    return task_info