from neutron_lib.api.definitions import default_subnetpools
from neutron_lib.tests.unit.api.definitions import base
class DefaultSubnetPoolsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = default_subnetpools
    extension_resources = ()
    extension_attributes = (default_subnetpools.USE_DEFAULT_SUBNETPOOL,)