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
def _format_resource_type(values):
    dt = timeutils.utcnow()
    resource_type = {'id': _get_metadef_id(), 'name': values['name'], 'protected': True, 'created_at': dt, 'updated_at': dt}
    resource_type.update(values)
    return resource_type