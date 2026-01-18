from neutron_lib.api.definitions import fip_distributed
from neutron_lib.api.definitions import l3
from neutron_lib.tests.unit.api.definitions import base
class FipDistributedDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = fip_distributed
    extension_resources = (l3.FLOATINGIPS,)
    extension_attributes = (fip_distributed.DISTRIBUTED,)