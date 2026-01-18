from neutron_lib.api.definitions import stateful_security_group
from neutron_lib.tests.unit.api.definitions import base
class StatefulSecurityGroupDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = stateful_security_group
    extension_attributes = ('stateful',)