import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
class TestServicePluginType(base.BaseTestCase):

    def setUp(self):
        super(TestServicePluginType, self).setUp()
        self._plugins = directory._PluginDirectory()
        self._plugins.add_plugin('stype', mock.Mock())
        self.useFixture(fixture.PluginDirectoryFixture(plugin_directory=self._plugins))

    def test_valid_plugin_type(self):
        self.assertIsNone(validators.validate_service_plugin_type('stype'))

    def test_invalid_plugin_type(self):
        self.assertRaisesRegex(n_exc.InvalidServiceType, 'Invalid service type', validators.validate_service_plugin_type, 'ntype')