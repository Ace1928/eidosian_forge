from neutron_lib.api.definitions import provider_net
from neutron_lib.tests.unit.api.definitions import base
class ProviderNetTestCase(base.DefinitionBaseTestCase):
    extension_module = provider_net
    extension_resources = (provider_net.COLLECTION_NAME,)
    extension_attributes = provider_net.ATTRIBUTES