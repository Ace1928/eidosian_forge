from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from glance.api import policy
from glance.common import wsgi
import glance.context
from glance.i18n import _, _LW
def _get_authenticated_context(self, req):
    service_catalog = None
    if req.headers.get('X-Service-Catalog') is not None:
        try:
            catalog_header = req.headers.get('X-Service-Catalog')
            service_catalog = jsonutils.loads(catalog_header)
        except ValueError:
            raise webob.exc.HTTPInternalServerError(_('Invalid service catalog json.'))
    request_id = req.headers.get('X-Openstack-Request-ID')
    if request_id and 0 < CONF.max_request_id_length < len(request_id):
        msg = _('x-openstack-request-id is too long, max size %s') % CONF.max_request_id_length
        return webob.exc.HTTPRequestHeaderFieldsTooLarge(comment=msg)
    kwargs = {'service_catalog': service_catalog, 'policy_enforcer': self.policy_enforcer, 'request_id': request_id}
    ctxt = glance.context.RequestContext.from_environ(req.environ, **kwargs)
    ctxt.roles = [r.lower() for r in ctxt.roles]
    return ctxt