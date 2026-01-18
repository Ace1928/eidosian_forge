from neutron_lib.api.definitions import bgpvpn
from neutron_lib.api.definitions import bgpvpn_routes_control
from neutron_lib.tests.unit.api.definitions import base
class BgpvpnRoutesControlDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = bgpvpn_routes_control
    extension_resources = (bgpvpn.COLLECTION_NAME,)
    extension_attributes = ('ports', 'routes', 'advertise_fixed_ips', 'advertise_extra_routes', 'local_pref')
    extension_subresources = ('port_associations', 'router_associations')