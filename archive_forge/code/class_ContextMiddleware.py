from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from glance.api import policy
from glance.common import wsgi
import glance.context
from glance.i18n import _, _LW
class ContextMiddleware(BaseContextMiddleware):

    def __init__(self, app):
        self.policy_enforcer = policy.Enforcer()
        super(ContextMiddleware, self).__init__(app)

    def process_request(self, req):
        """Convert authentication information into a request context

        Generate a glance.context.RequestContext object from the available
        authentication headers and store on the 'context' attribute
        of the req object.

        :param req: wsgi request object that will be given the context object
        :raises webob.exc.HTTPUnauthorized: when value of the
                                            X-Identity-Status  header is not
                                            'Confirmed' and anonymous access
                                            is disallowed
        """
        if req.headers.get('X-Identity-Status') == 'Confirmed':
            req.context = self._get_authenticated_context(req)
        elif CONF.allow_anonymous_access:
            req.context = self._get_anonymous_context()
        else:
            raise webob.exc.HTTPUnauthorized()

    def _get_anonymous_context(self):
        kwargs = {'user': None, 'tenant': None, 'roles': [], 'is_admin': False, 'read_only': True, 'policy_enforcer': self.policy_enforcer}
        return glance.context.RequestContext(**kwargs)

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