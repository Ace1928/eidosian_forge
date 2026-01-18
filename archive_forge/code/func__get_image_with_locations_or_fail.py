import hashlib
import json
from oslo_utils import encodeutils
from requests import codes
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import schemas
def _get_image_with_locations_or_fail(self, image_id):
    image = self.get(image_id)
    if getattr(image, 'locations', None) is None:
        raise exc.HTTPBadRequest('The administrator has disabled API access to image locations')
    return image