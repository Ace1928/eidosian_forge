from neutron_lib.api.definitions import port_device_profile
from neutron_lib.tests.unit.api.definitions import base
class PortDeviceProfileDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_device_profile
    extension_resources = (port_device_profile.COLLECTION_NAME,)
    extension_attributes = (port_device_profile.DEVICE_PROFILE,)