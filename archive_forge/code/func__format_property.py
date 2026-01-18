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
def _format_property(values):
    property = {'id': _get_metadef_id(), 'namespace_id': None, 'name': None, 'json_schema': None}
    property.update(values)
    return property