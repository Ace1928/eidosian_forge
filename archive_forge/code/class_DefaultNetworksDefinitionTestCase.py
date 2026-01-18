from neutron_lib.api.definitions import project_default_networks as dn
from neutron_lib.tests.unit.api.definitions import base
class DefaultNetworksDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dn
    extension_attributes = (dn.PROJECT_DEFAULT,)