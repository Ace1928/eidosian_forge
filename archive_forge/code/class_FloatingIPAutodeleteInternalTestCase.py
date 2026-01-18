from neutron_lib.api.definitions import floatingip_autodelete_internal
from neutron_lib.tests.unit.api.definitions import base
class FloatingIPAutodeleteInternalTestCase(base.DefinitionBaseTestCase):
    extension_module = floatingip_autodelete_internal