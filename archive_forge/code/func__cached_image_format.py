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
def _cached_image_format(cached_image):
    """Format a cached image for consumption outside of this module"""
    image_dict = {'id': cached_image['id'], 'image_id': cached_image['image_id'], 'last_accessed': cached_image['last_accessed'].timestamp(), 'last_modified': cached_image['last_modified'].timestamp(), 'size': cached_image['size'], 'hits': cached_image['hits'], 'checksum': cached_image['checksum']}
    return image_dict