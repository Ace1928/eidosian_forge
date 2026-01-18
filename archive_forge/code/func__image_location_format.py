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
@utils.no_4byte_params
def _image_location_format(image_id, value, meta_data, status, deleted=False):
    dt = timeutils.utcnow()
    return {'id': str(uuid.uuid4()), 'image_id': image_id, 'created_at': dt, 'updated_at': dt, 'deleted_at': dt if deleted else None, 'deleted': deleted, 'url': value, 'metadata': meta_data, 'status': status}