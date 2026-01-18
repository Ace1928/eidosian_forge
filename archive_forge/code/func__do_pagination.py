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
def _do_pagination(context, images, marker, limit, show_deleted, status='accepted'):
    start = 0
    end = -1
    if marker is None:
        start = 0
    else:
        _image_get(context, marker, force_show_deleted=show_deleted, status=status)
        for i, image in enumerate(images):
            if image['id'] == marker:
                start = i + 1
                break
        else:
            raise exception.ImageNotFound()
    end = start + limit if limit is not None else None
    return images[start:end]