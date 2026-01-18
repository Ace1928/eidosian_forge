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
def _image_get(context, image_id, force_show_deleted=False, status=None):
    try:
        image = DATA['images'][image_id]
    except KeyError:
        LOG.warning(_LW('Could not find image %s'), image_id)
        raise exception.ImageNotFound()
    if image['deleted'] and (not (force_show_deleted or context.can_see_deleted)):
        LOG.warning(_LW('Unable to get deleted image'))
        raise exception.ImageNotFound()
    if not is_image_visible(context, image):
        LOG.warning(_LW('Unable to get unowned image'))
        raise exception.Forbidden('Image not visible to you')
    return image