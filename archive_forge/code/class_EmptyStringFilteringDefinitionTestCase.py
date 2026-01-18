from neutron_lib.api.definitions import empty_string_filtering
from neutron_lib.tests.unit.api.definitions import base
class EmptyStringFilteringDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = empty_string_filtering