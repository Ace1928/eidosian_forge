from neutron_lib.api.definitions import rbac_bgpvpn
from neutron_lib import constants
from neutron_lib.tests.unit.api.definitions import base
class RbacBGPVPNDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = rbac_bgpvpn
    extension_resources = (rbac_bgpvpn.COLLECTION_NAME,)
    extension_attributes = (constants.SHARED,)