from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
class OctaviaClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = octavia
    service_types = [LOADBALANCER] = ['load-balancer']
    supported_versions = [V2] = ['2']
    default_version = V2

    def _create(self, version=None):
        interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        endpoint = self.url_for(service_type=self.LOADBALANCER, endpoint_type=interface)
        return octavia.OctaviaAPI(session=self.context.keystone_session, service_type=self.LOADBALANCER, endpoint=endpoint)

    def is_not_found(self, ex):
        return isinstance(ex, exceptions.NotFound) or _is_translated_exception(ex, 404)

    def is_over_limit(self, ex):
        return isinstance(ex, exceptions.OverLimit) or _is_translated_exception(ex, 413)

    def is_conflict(self, ex):
        return isinstance(ex, exceptions.Conflict) or _is_translated_exception(ex, 409)

    def get_pool(self, value):
        pool = self.client().find(path=constants.BASE_POOL_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return pool['id']

    def get_listener(self, value):
        lsnr = self.client().find(path=constants.BASE_LISTENER_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return lsnr['id']

    def get_loadbalancer(self, value):
        lb = self.client().find(path=constants.BASE_LOADBALANCER_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return lb['id']

    def get_l7policy(self, value):
        policy = self.client().find(path=constants.BASE_L7POLICY_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return policy['id']

    def get_flavor(self, value):
        flavor = self.client().find(path=constants.BASE_FLAVOR_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return flavor['id']

    def get_flavorprofile(self, value):
        flavorprofile = self.client().find(path=constants.BASE_FLAVORPROFILE_URL, value=value, attr=DEFAULT_FIND_ATTR)
        return flavorprofile['id']