from neutron_lib.api.definitions import bgp
from neutron_lib.tests.unit.api.definitions import base
class BgpDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = bgp
    extension_resources = ('bgp-speakers', 'bgp-peers')
    extension_attributes = ('local_as', 'peers', 'networks', 'advertise_floating_ip_host_routes', 'advertise_tenant_networks', 'peer_ip', 'remote_as', 'auth_type', 'password')