from neutron_lib.api.definitions import vlantransparent
from neutron_lib.tests.unit.api.definitions import base
class VlantransparentDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = vlantransparent
    extension_resources = ()
    extension_attributes = (vlantransparent.VLANTRANSPARENT,)

    def test_get_vlan_transparent(self):
        self.assertTrue(vlantransparent.get_vlan_transparent({vlantransparent.VLANTRANSPARENT: True, 'vlan': '1'}))

    def test_get_vlan_transparent_not_set(self):
        self.assertFalse(vlantransparent.get_vlan_transparent({'vlanxtx': True, 'vlan': '1'}))