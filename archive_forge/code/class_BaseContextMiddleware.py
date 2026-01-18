from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from glance.api import policy
from glance.common import wsgi
import glance.context
from glance.i18n import _, _LW
class BaseContextMiddleware(wsgi.Middleware):

    def process_response(self, resp):
        try:
            request_id = resp.request.context.request_id
        except AttributeError:
            LOG.warning(_LW('Unable to retrieve request id from context'))
        else:
            prefix = b'req-' if isinstance(request_id, bytes) else 'req-'
            if not request_id.startswith(prefix):
                request_id = prefix + request_id
            resp.headers['x-openstack-request-id'] = request_id
        return resp