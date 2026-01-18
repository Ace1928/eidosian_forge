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
def _image_property_format(image_id, name, value):
    return {'image_id': image_id, 'name': name, 'value': value, 'deleted': False, 'deleted_at': None}