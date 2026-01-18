import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def _image_present(client, image_uuid):
    """Check if an image is present in glance.

    client: the ImageService
    image_uuid: the image uuid to check

    Returns: True if the image is present
    """
    headers = client.get_image_meta(image_uuid)
    return 'status' in headers