from neutron_lib.api.definitions import subnetpool
from neutron_lib.tests.unit.api.definitions import base
class SubnetPoolDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnetpool
    extension_attributes = ()