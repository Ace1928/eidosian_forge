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
def _get_v2_image_metadata(self, request, image_id):
    """
        Retrieves image and for v2 api and creates adapter like object
        to access image core or custom properties on request.
        """
    db_api = glance.db.get_api()
    image_repo = glance.db.ImageRepo(request.context, db_api)
    try:
        image = image_repo.get(image_id)
        request.environ['api.cache.image'] = image
        return (image, policy.ImageTarget(image))
    except exception.NotFound as e:
        raise webob.exc.HTTPNotFound(explanation=e.msg, request=request)