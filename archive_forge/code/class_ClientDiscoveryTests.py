from oslo_serialization import jsonutils
from keystoneclient.generic import client
from keystoneclient.tests.unit import utils
class ClientDiscoveryTests(utils.TestCase):

    def test_discover_extensions_v2(self):
        self.requests_mock.get('%s/extensions' % V2_URL, text=EXTENSION_LIST)
        with self.deprecations.expect_deprecations_here():
            extensions = client.Client().discover_extensions(url=V2_URL)
        self.assertIn(EXTENSION_ALIAS_FOO, extensions)
        self.assertEqual(extensions[EXTENSION_ALIAS_FOO], EXTENSION_NAME_FOO)
        self.assertIn(EXTENSION_ALIAS_BAR, extensions)
        self.assertEqual(extensions[EXTENSION_ALIAS_BAR], EXTENSION_NAME_BAR)