from neutron_lib.api.definitions import l3_enable_default_route_bfd
from neutron_lib.tests.unit.api.definitions import base
class L3EnableDefaultRouteBFDDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_enable_default_route_bfd
    extension_resources = ()
    extension_attributes = (l3_enable_default_route_bfd.ENABLE_DEFAULT_ROUTE_BFD,)