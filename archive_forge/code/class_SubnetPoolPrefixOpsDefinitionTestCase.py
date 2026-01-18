from neutron_lib.api.definitions import subnetpool_prefix_ops
from neutron_lib.tests.unit.api.definitions import base
class SubnetPoolPrefixOpsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnetpool_prefix_ops
    extension_attributes = ()