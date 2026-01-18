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
def _process_GET_response(self, resp, image_id, version=None):
    image_checksum = resp.headers.get('Content-MD5')
    if not image_checksum:
        image_checksum = resp.headers.get('x-image-meta-checksum')
    if not image_checksum:
        LOG.error(_LE('Checksum header is missing.'))
    image = None
    if version:
        method = getattr(self, '_get_%s_image_metadata' % version)
        image, metadata = method(resp.request, image_id)
    self._enforce(resp.request, image)
    resp.app_iter = self.cache.get_caching_iter(image_id, image_checksum, resp.app_iter)
    return resp