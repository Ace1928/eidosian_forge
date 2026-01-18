from neutron_lib.api.definitions import subnet_onboard
from neutron_lib.tests.unit.api.definitions import base
class OnboardSubnetDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet_onboard
    extension_attributes = ()