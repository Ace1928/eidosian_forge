from oslo_messaging import exceptions
from webob import exc
from heat.api.openstack.v1 import util
from heat.common.i18n import _
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import client as rpc_client
class ServiceController(object):
    """WSGI controller for reporting the heat engine status in Heat v1 API."""
    REQUEST_SCOPE = 'service'

    def __init__(self, options):
        self.options = options
        self.rpc_client = rpc_client.EngineClient()

    @util.registered_policy_enforce
    def index(self, req):
        try:
            services = self.rpc_client.list_services(req.context)
            return {'services': services}
        except exceptions.MessagingTimeout:
            msg = _('All heat engines are down.')
            raise exc.HTTPServiceUnavailable(msg)