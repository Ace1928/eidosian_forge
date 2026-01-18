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
def _verify_metadata(self, image_meta):
    """
        Sanity check the 'deleted' and 'size' metadata values.
        """
    if image_meta['status'] == 'deleted' and image_meta['deleted']:
        raise exception.NotFound()
    if not image_meta['size']:
        if not isinstance(image_meta, policy.ImageTarget):
            image_meta['size'] = self.cache.get_image_size(image_meta['id'])
        else:
            image_meta.target.size = self.cache.get_image_size(image_meta['id'])