import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
import urllib.parse
from glanceclient.common import utils
from glanceclient.v1.apiclient import base
def _image_meta_to_headers(self, fields):
    headers = {}
    fields_copy = copy.deepcopy(fields)

    def to_str(value):
        if not isinstance(value, str):
            return str(value)
        return value
    for key, value in fields_copy.pop('properties', {}).items():
        headers['x-image-meta-property-%s' % key] = to_str(value)
    for key, value in fields_copy.items():
        headers['x-image-meta-%s' % key] = to_str(value)
    return headers