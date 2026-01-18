from neutron_lib.api.definitions import l2_adjacency
from neutron_lib.tests.unit.api.definitions import base
class L2AdjacencyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l2_adjacency
    extension_resources = ()
    extension_attributes = (l2_adjacency.L2_ADJACENCY,)