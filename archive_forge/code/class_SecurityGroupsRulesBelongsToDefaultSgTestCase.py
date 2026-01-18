from neutron_lib.api.definitions import \
from neutron_lib.tests.unit.api.definitions import base
class SecurityGroupsRulesBelongsToDefaultSgTestCase(base.DefinitionBaseTestCase):
    extension_module = security_groups_rules_belongs_to_default_sg
    extension_resources = ('security_group_rules',)
    extension_attributes = ('belongs_to_default_sg',)