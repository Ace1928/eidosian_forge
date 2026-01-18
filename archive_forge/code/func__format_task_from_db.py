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
def _format_task_from_db(task_ref, task_info_ref):
    task = copy.deepcopy(task_ref)
    if task_info_ref:
        task_info = copy.deepcopy(task_info_ref)
        task_info_values = _pop_task_info_values(task_info)
        task.update(task_info_values)
    return task