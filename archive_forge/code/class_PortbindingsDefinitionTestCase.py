from neutron_lib.api.definitions import portbindings
from neutron_lib.tests.unit.api.definitions import base
class PortbindingsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = portbindings
    extension_resources = (portbindings.COLLECTION_NAME,)
    extension_attributes = (portbindings.VIF_TYPE, portbindings.VIF_DETAILS, portbindings.VNIC_TYPE, portbindings.HOST_ID, portbindings.PROFILE)