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
def _task_info_format(task_id, **values):
    task_info = {'task_id': task_id, 'input': None, 'result': None, 'message': None}
    task_info.update(values)
    return task_info