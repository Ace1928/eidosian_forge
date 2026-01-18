from neutron_lib.api.definitions import ip_allocation
from neutron_lib.tests.unit.api.definitions import base
class IPAllocationDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = ip_allocation
    extension_resources = ()
    extension_attributes = (ip_allocation.IP_ALLOCATION,)