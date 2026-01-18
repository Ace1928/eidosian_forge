import copy
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
def _lookup_image(self, req, image_id):
    image_repo = self.gateway.get_repo(req.context)
    try:
        return image_repo.get(image_id)
    except exception.NotFound:
        msg = _('Image %s not found.') % image_id
        LOG.warning(msg)
        raise webob.exc.HTTPNotFound(explanation=msg)
    except exception.Forbidden:
        msg = _('You are not authorized to lookup image %s.') % image_id
        LOG.warning(msg)
        raise webob.exc.HTTPForbidden(explanation=msg)