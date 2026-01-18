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
def _image_member_format(image_id, tenant_id, can_share, status='pending', deleted=False):
    dt = timeutils.utcnow()
    return {'id': str(uuid.uuid4()), 'image_id': image_id, 'member': tenant_id, 'can_share': can_share, 'status': status, 'created_at': dt, 'updated_at': dt, 'deleted': deleted}