from neutron_lib.api.definitions import taas
from neutron_lib.tests.unit.api.definitions import base
class TaasDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = taas
    extension_resources = (taas.COLLECTION_NAME, taas.TAP_FLOWS)
    extension_attributes = ('port_id', 'tap_service_id', 'source_port', 'direction')