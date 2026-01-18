from neutron_lib.api.definitions import fip_pf_description as pf_description
from neutron_lib.api.definitions import floating_ip_port_forwarding as fip_pf
from neutron_lib.tests.unit.api.definitions import base
class FipPortForwardingNameAndDescriptionTestCase(base.DefinitionBaseTestCase):
    extension_module = pf_description
    extension_subresources = (fip_pf.COLLECTION_NAME,)
    extension_attributes = (pf_description.DESCRIPTION_FIELD,)