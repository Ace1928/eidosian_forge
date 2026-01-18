from neutron_lib.api.definitions import servicetype
from neutron_lib.tests.unit.api.definitions import base
class ServiceTypeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = servicetype
    extension_resources = (servicetype.COLLECTION_NAME,)
    extension_attributes = ('default', servicetype.SERVICE_ATTR)