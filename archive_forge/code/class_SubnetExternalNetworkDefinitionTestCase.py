from neutron_lib.api.definitions import subnet_external_network
from neutron_lib.tests.unit.api.definitions import base
class SubnetExternalNetworkDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet_external_network
    extension_attributes = (subnet_external_network.RESOURCE_NAME,)