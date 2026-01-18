from neutron_lib.api.definitions import expose_port_forwarding_in_fip
from neutron_lib.api.definitions import floating_ip_port_forwarding
from neutron_lib.tests.unit.api.definitions import base
class ExposePortForwardingInFip(base.DefinitionBaseTestCase):
    extension_module = expose_port_forwarding_in_fip
    extension_resources = (expose_port_forwarding_in_fip.COLLECTION_NAME,)
    extension_attributes = (floating_ip_port_forwarding.COLLECTION_NAME,)