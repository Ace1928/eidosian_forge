from neutron_lib.api.definitions import router_interface_fip
from neutron_lib.tests.unit.api.definitions import base
class RouterInterfaceFipDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = router_interface_fip
    extension_resources = ()
    extension_attributes = ()