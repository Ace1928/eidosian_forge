from neutron_lib.api.definitions import security_groups_normalized_cidr
from neutron_lib.tests.unit.api.definitions import base
class SecurityGroupsNormalizedCidrDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = security_groups_normalized_cidr
    extension_resources = ('security_group_rules',)
    extension_attributes = ('normalized_cidr',)