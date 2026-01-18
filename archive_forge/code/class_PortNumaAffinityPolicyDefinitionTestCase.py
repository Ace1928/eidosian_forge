from neutron_lib.api.definitions import port_numa_affinity_policy
from neutron_lib.tests.unit.api.definitions import base
class PortNumaAffinityPolicyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_numa_affinity_policy
    extension_resources = (port_numa_affinity_policy.COLLECTION_NAME,)
    extension_attributes = (port_numa_affinity_policy.NUMA_AFFINITY_POLICY,)