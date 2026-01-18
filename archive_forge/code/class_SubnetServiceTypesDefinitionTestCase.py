from neutron_lib.api.definitions import subnet as subnet_def
from neutron_lib.api.definitions import subnet_service_types
from neutron_lib.tests.unit.api.definitions import base
class SubnetServiceTypesDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet_service_types
    extension_resource = (subnet_def.RESOURCE_NAME,)
    extension_attributes = ()