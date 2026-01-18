from neutron_lib.api.definitions import l3_conntrack_helper
from neutron_lib.tests.unit.api.definitions import base
class ConntrackHelperDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_conntrack_helper
    extension_resources = (l3_conntrack_helper.PARENT_COLLECTION_NAME,)
    extension_attributes = (l3_conntrack_helper.ID, l3_conntrack_helper.PROTOCOL, l3_conntrack_helper.PORT, l3_conntrack_helper.HELPER, l3_conntrack_helper.PROJECT_ID)
    extension_subresources = (l3_conntrack_helper.COLLECTION_NAME,)