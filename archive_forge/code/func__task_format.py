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
def _task_format(task_id, **values):
    dt = timeutils.utcnow()
    task = {'id': task_id, 'type': 'import', 'status': values.get('status', 'pending'), 'owner': None, 'expires_at': None, 'created_at': dt, 'updated_at': dt, 'deleted_at': None, 'deleted': False, 'image_id': values.get('image_id', None), 'request_id': values.get('request_id', None), 'user_id': values.get('user_id', None)}
    task.update(values)
    return task