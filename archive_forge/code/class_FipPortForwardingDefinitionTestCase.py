from neutron_lib.api.definitions import floating_ip_port_forwarding as fip_pf
from neutron_lib.tests.unit.api.definitions import base
class FipPortForwardingDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = fip_pf
    extension_resources = (fip_pf.PARENT_COLLECTION_NAME,)
    extension_attributes = (fip_pf.ID, fip_pf.PROJECT_ID, fip_pf.EXTERNAL_PORT, fip_pf.INTERNAL_PORT, fip_pf.INTERNAL_IP_ADDRESS, fip_pf.PROTOCOL, fip_pf.INTERNAL_PORT_ID)
    extension_subresources = (fip_pf.COLLECTION_NAME,)