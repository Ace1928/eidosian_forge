import hashlib
import json
from oslo_utils import encodeutils
from requests import codes
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import schemas
@utils.add_req_id_to_object()
def _send_image_update_request(self, image_id, patch_body):
    url = '/v2/images/%s' % image_id
    hdrs = {'Content-Type': 'application/openstack-images-v2.1-json-patch'}
    resp, body = self.http_client.patch(url, headers=hdrs, data=json.dumps(patch_body))
    return ((resp, body), resp)