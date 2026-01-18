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
def _sort_tasks(tasks, sort_key, sort_dir):
    reverse = False
    if tasks and (not sort_key in tasks[0]):
        raise exception.InvalidSortKey()

    def keyfn(x):
        return (x[sort_key] if x[sort_key] is not None else '', x['created_at'], x['id'])
    reverse = sort_dir == 'desc'
    tasks.sort(key=keyfn, reverse=reverse)
    return tasks