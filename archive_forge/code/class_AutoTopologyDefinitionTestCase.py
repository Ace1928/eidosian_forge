from neutron_lib.api.definitions import auto_allocated_topology
from neutron_lib.tests.unit.api.definitions import base
class AutoTopologyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = auto_allocated_topology
    extension_resources = (auto_allocated_topology.COLLECTION_NAME,)
    extension_attributes = ()