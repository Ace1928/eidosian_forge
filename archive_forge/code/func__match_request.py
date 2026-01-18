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
@staticmethod
def _match_request(request):
    """Determine the version of the url and extract the image id

        :returns: tuple of version and image id if the url is a cacheable,
                 otherwise None
        """
    for (version, method), pattern in PATTERNS.items():
        if request.method != method:
            continue
        match = pattern.match(request.path_info)
        if match is None:
            continue
        image_id = match.group(1)
        if image_id != 'detail':
            return (version, method, image_id)