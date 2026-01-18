import os
from openstackclient.common import configuration
from openstackclient.tests.functional import base
class ConfigurationTests(base.TestCase):
    """Functional test for configuration."""

    def test_configuration_show(self):
        raw_output = self.openstack('configuration show')
        items = self.parse_listing(raw_output)
        self.assert_table_structure(items, BASIC_CONFIG_HEADERS)
        cmd_output = self.openstack('configuration show', parse_output=True)
        self.assertEqual(configuration.REDACTED, cmd_output['auth.password'])
        self.assertIn('auth.password', cmd_output.keys())
        cmd_output = self.openstack('configuration show --mask', parse_output=True)
        self.assertEqual(configuration.REDACTED, cmd_output['auth.password'])
        cmd_output = self.openstack('configuration show --unmask', parse_output=True)
        passwd = os.environ.get('OS_PASSWORD')
        if passwd:
            self.assertEqual(passwd, cmd_output['auth.password'])
        else:
            self.assertNotEqual(configuration.REDACTED, cmd_output['auth.password'])