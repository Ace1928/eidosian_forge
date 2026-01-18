from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
class TestExtensionIsSupported(base.BaseTestCase):

    def setUp(self):
        super(TestExtensionIsSupported, self).setUp()
        self._plugin = DummyPlugin()

    def test_extension_exists(self):
        self.assertTrue(extensions.is_extension_supported(self._plugin, 'flash'))

    def test_extension_does_not_exist(self):
        self.assertFalse(extensions.is_extension_supported(self._plugin, 'gordon'))