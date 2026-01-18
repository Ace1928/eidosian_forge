import http.client as http
import re
from oslo_log import log as logging
import webob
from glance.api.common import size_checked_iter
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
import glance.db
from glance.i18n import _LE, _LI
from glance import image_cache
from glance import notifier
def _process_v2_request(self, request, image_id, image_iterator, image_meta):
    image = request.environ['api.cache.image']
    self._verify_metadata(image_meta)
    response = webob.Response(request=request)
    response.app_iter = size_checked_iter(response, image_meta, image_meta['size'], image_iterator, notifier.Notifier())
    response.headers['Content-Type'] = 'application/octet-stream'
    if image.checksum:
        response.headers['Content-MD5'] = image.checksum
    response.headers['Content-Length'] = str(image.size)
    return response