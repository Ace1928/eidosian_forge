from neutron_lib.api.definitions import l3_flavors
from neutron_lib.tests.unit.api.definitions import base
class L3FlavorsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_flavors
    extension_resources = ()
    extension_attributes = (l3_flavors.FLAVOR_ID,)