from neutron_lib.api.definitions import port_resource_request
from neutron_lib.tests.unit.api.definitions import base
class PortResourceRequestDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = port_resource_request
    extension_attributes = (port_resource_request.RESOURCE_REQUEST,)