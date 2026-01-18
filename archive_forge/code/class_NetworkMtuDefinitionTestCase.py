from neutron_lib.api.definitions import network_mtu
from neutron_lib.tests.unit.api.definitions import base
class NetworkMtuDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network_mtu
    extension_attributes = ('mtu',)