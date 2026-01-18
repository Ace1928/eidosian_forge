from neutron_lib.api.definitions import trunk
from neutron_lib.tests.unit.api.definitions import base
class TrunkDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = trunk
    extension_resources = (trunk.TRUNKS,)
    extension_attributes = (trunk.SUB_PORTS,)