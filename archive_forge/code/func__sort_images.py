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
def _sort_images(images, sort_key, sort_dir):
    sort_key = ['created_at'] if not sort_key else sort_key
    default_sort_dir = 'desc'
    if not sort_dir:
        sort_dir = [default_sort_dir] * len(sort_key)
    elif len(sort_dir) == 1:
        default_sort_dir = sort_dir[0]
        sort_dir *= len(sort_key)
    for key in ['created_at', 'id']:
        if key not in sort_key:
            sort_key.append(key)
            sort_dir.append(default_sort_dir)
    for key in sort_key:
        if images and (not key in images[0]):
            raise exception.InvalidSortKey()
    if any((dir for dir in sort_dir if dir not in ['asc', 'desc'])):
        raise exception.InvalidSortDir()
    if len(sort_key) != len(sort_dir):
        raise exception.Invalid(message='Number of sort dirs does not match the number of sort keys')
    for key, dir in reversed(list(zip(sort_key, sort_dir))):
        reverse = dir == 'desc'
        images.sort(key=lambda x: x[key] or '', reverse=reverse)
    return images