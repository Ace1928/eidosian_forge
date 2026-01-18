from neutron_lib.api.definitions import _dummy
from neutron_lib.tests.unit.api.definitions import base
class _DummyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = _dummy
    extension_resources = (_dummy.COLLECTION_NAME,)
    extension_subresources = ('subfoo',)