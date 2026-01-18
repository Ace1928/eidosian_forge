from neutron_lib.api.definitions import security_groups_remote_address_group
from neutron_lib.tests.unit.api.definitions import base
class SecurityGroupsRemoteAddressGroupDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = security_groups_remote_address_group
    extension_resources = ('security_group_rules',)
    extension_attributes = ('remote_address_group_id',)