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
def _image_update(image, values, properties):
    properties = [{'name': k, 'value': v, 'image_id': image['id'], 'deleted': False} for k, v in properties.items()]
    if 'properties' not in image.keys():
        image['properties'] = []
    image['properties'].extend(properties)
    values = db_utils.ensure_image_dict_v2_compliant(values)
    image.update(values)
    return image