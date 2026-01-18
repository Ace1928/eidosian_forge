from neutron_lib.api.definitions import port_numa_affinity_policy_socket
from neutron_lib.tests.unit.api.definitions import base
class PortNumaAffinityPolicySocketDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_numa_affinity_policy_socket
    extension_resources = (port_numa_affinity_policy_socket.COLLECTION_NAME,)
    extension_attributes = (port_numa_affinity_policy_socket.NUMA_AFFINITY_POLICY,)