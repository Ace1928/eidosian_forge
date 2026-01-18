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
def _format_object(values):
    dt = timeutils.utcnow()
    object = {'id': _get_metadef_id(), 'namespace_id': None, 'name': None, 'description': None, 'json_schema': None, 'required': None, 'created_at': dt, 'updated_at': dt}
    object.update(values)
    return object