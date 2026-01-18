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
def _task_soft_delete(context):
    """Scrub task entities which are expired """
    global DATA
    now = timeutils.utcnow()
    tasks = DATA['tasks'].values()
    for task in tasks:
        if task['owner'] == context.owner and task['deleted'] == False and (task['expires_at'] <= now):
            task['deleted'] = True
            task['deleted_at'] = timeutils.utcnow()