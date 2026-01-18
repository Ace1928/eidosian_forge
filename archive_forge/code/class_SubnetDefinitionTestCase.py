from neutron_lib.api.definitions import subnet
from neutron_lib.tests.unit.api.definitions import base
class SubnetDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet
    extension_attributes = ()