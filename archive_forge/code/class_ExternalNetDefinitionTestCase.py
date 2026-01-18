from neutron_lib.api.definitions import external_net
from neutron_lib.tests.unit.api.definitions import base
class ExternalNetDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = external_net
    extension_resources = ()
    extension_attributes = (external_net.EXTERNAL,)