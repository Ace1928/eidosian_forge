from neutron_lib.api.definitions import security_groups_shared_filtering
from neutron_lib import constants
from neutron_lib.tests.unit.api.definitions import base
class SecurityGroupsSharedFilteringDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = security_groups_shared_filtering
    extension_resources = ('security_groups',)
    extension_attributes = (constants.SHARED,)