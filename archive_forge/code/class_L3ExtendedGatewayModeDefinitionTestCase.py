from neutron_lib.api.definitions import l3_ext_gw_mode
from neutron_lib.tests.unit.api.definitions import base
class L3ExtendedGatewayModeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_ext_gw_mode
    extension_attributes = ('external_gateway_info',)