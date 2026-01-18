from neutron_lib.api.definitions import taas
from neutron_lib.api.definitions import vlan_filter
from neutron_lib.tests.unit.api.definitions import base
class TaasVlanFilterDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = vlan_filter
    extension_resources = (taas.TAP_FLOWS,)
    extension_attributes = ('vlan_filter',)