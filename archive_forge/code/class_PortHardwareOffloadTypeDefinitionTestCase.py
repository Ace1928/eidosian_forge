from neutron_lib.api.definitions import port_hardware_offload_type
from neutron_lib.tests.unit.api.definitions import base
class PortHardwareOffloadTypeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_hardware_offload_type
    extension_resources = (port_hardware_offload_type.COLLECTION_NAME,)
    extension_attributes = (port_hardware_offload_type.HARDWARE_OFFLOAD_TYPE,)