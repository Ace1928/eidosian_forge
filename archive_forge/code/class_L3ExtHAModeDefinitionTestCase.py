from neutron_lib.api.definitions import l3_ext_ha_mode
from neutron_lib.tests.unit.api.definitions import base
class L3ExtHAModeDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_ext_ha_mode
    extension_resources = ()
    extension_attributes = (l3_ext_ha_mode.HA_INFO,)