from neutron_lib.api.definitions import fip_port_details
from neutron_lib.api.definitions import l3
from neutron_lib.tests.unit.api.definitions import base
class FipPortDetailsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = fip_port_details
    extension_resources = (l3.FLOATINGIPS,)
    extension_attributes = (fip_port_details.PORT_DETAILS,)