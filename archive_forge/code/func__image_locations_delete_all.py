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
def _image_locations_delete_all(context, image_id, delete_time=None):
    image = DATA['images'][image_id]
    for loc in image['locations']:
        if not loc['deleted']:
            image_location_delete(context, image_id, loc['id'], 'deleted', delete_time=delete_time)
    for i, loc in enumerate(DATA['locations']):
        if image_id == loc['image_id'] and loc['deleted'] == False:
            del DATA['locations'][i]