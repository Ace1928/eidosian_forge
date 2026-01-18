from neutron_lib.api.definitions import fip_pf_port_range as pf_port_range
from neutron_lib.api.definitions import floating_ip_port_forwarding as fip_pf
from neutron_lib.tests.unit.api.definitions import base
class FipPortForwardingPortRangeTestCase(base.DefinitionBaseTestCase):
    extension_module = pf_port_range
    extension_subresources = (fip_pf.COLLECTION_NAME,)
    extension_attributes = (pf_port_range.EXTERNAL_PORT_RANGE, pf_port_range.INTERNAL_PORT_RANGE, fip_pf.EXTERNAL_PORT, fip_pf.INTERNAL_PORT)