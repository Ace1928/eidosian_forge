import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
class TestShellCli(utils.TestShell):
    """Test handling of specific global options

    _shell.options is the parsed command line from argparse
    _shell.client_manager.* are the values actually used

    """

    def setUp(self):
        super(TestShellCli, self).setUp()
        env = {}
        self.useFixture(utils.EnvFixture(env.copy()))

    def test_shell_args_no_options(self):
        _shell = utils.make_shell()
        with mock.patch('osc_lib.shell.OpenStackShell.initialize_app', self.app):
            utils.fake_execute(_shell, 'list user')
            self.app.assert_called_with(['list', 'user'])

    def test_shell_args_tls_options(self):
        """Test the TLS verify and CA cert file options"""
        _shell = utils.make_shell()
        utils.fake_execute(_shell, 'module list')
        self.assertIsNone(_shell.options.verify)
        self.assertIsNone(_shell.options.insecure)
        self.assertIsNone(_shell.options.cacert)
        self.assertTrue(_shell.client_manager.verify)
        self.assertIsNone(_shell.client_manager.cacert)
        utils.fake_execute(_shell, '--verify module list')
        self.assertTrue(_shell.options.verify)
        self.assertIsNone(_shell.options.insecure)
        self.assertIsNone(_shell.options.cacert)
        self.assertTrue(_shell.client_manager.verify)
        self.assertIsNone(_shell.client_manager.cacert)
        utils.fake_execute(_shell, '--insecure module list')
        self.assertIsNone(_shell.options.verify)
        self.assertTrue(_shell.options.insecure)
        self.assertIsNone(_shell.options.cacert)
        self.assertFalse(_shell.client_manager.verify)
        self.assertIsNone(_shell.client_manager.cacert)
        utils.fake_execute(_shell, '--os-cacert foo module list')
        self.assertIsNone(_shell.options.verify)
        self.assertIsNone(_shell.options.insecure)
        self.assertEqual('foo', _shell.options.cacert)
        self.assertEqual('foo', _shell.client_manager.verify)
        self.assertEqual('foo', _shell.client_manager.cacert)
        utils.fake_execute(_shell, '--os-cacert foo --verify module list')
        self.assertTrue(_shell.options.verify)
        self.assertIsNone(_shell.options.insecure)
        self.assertEqual('foo', _shell.options.cacert)
        self.assertEqual('foo', _shell.client_manager.verify)
        self.assertEqual('foo', _shell.client_manager.cacert)
        utils.fake_execute(_shell, '--os-cacert foo --insecure module list')
        self.assertIsNone(_shell.options.verify)
        self.assertTrue(_shell.options.insecure)
        self.assertEqual('foo', _shell.options.cacert)
        self.assertFalse(_shell.client_manager.verify)
        self.assertIsNone(_shell.client_manager.cacert)

    def test_shell_args_cert_options(self):
        """Test client cert options"""
        _shell = utils.make_shell()
        utils.fake_execute(_shell, 'module list')
        self.assertEqual('', _shell.options.cert)
        self.assertEqual('', _shell.options.key)
        self.assertIsNone(_shell.client_manager.cert)
        utils.fake_execute(_shell, '--os-cert mycert module list')
        self.assertEqual('mycert', _shell.options.cert)
        self.assertEqual('', _shell.options.key)
        self.assertEqual('mycert', _shell.client_manager.cert)
        utils.fake_execute(_shell, '--os-key mickey module list')
        self.assertEqual('', _shell.options.cert)
        self.assertEqual('mickey', _shell.options.key)
        self.assertIsNone(_shell.client_manager.cert)
        utils.fake_execute(_shell, '--os-cert mycert --os-key mickey module list')
        self.assertEqual('mycert', _shell.options.cert)
        self.assertEqual('mickey', _shell.options.key)
        self.assertEqual(('mycert', 'mickey'), _shell.client_manager.cert)

    @mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
    def test_shell_args_cloud_no_vendor(self, config_mock):
        """Test cloud config options without the vendor file"""
        config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_1))
        _shell = utils.make_shell()
        utils.fake_execute(_shell, '--os-cloud scc module list')
        self.assertEqual('scc', _shell.cloud.name)
        self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
        self.assertEqual(DEFAULT_PROJECT_NAME, _shell.cloud.config['auth']['project_name'])
        self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
        self.assertEqual('occ-cloud', _shell.cloud.config['region_name'])
        self.assertEqual('occ-cloud', _shell.client_manager.region_name)
        self.assertEqual('glazed', _shell.cloud.config['donut'])
        self.assertEqual('admin', _shell.cloud.config['interface'])
        self.assertIsNone(_shell.cloud.config['cert'])
        self.assertIsNone(_shell.cloud.config['key'])
        self.assertIsNone(_shell.client_manager.cert)

    @mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
    @mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
    def test_shell_args_cloud_public(self, config_mock, public_mock):
        """Test cloud config options with the vendor file"""
        config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_2))
        public_mock.return_value = ('file.yaml', copy.deepcopy(PUBLIC_1))
        _shell = utils.make_shell()
        utils.fake_execute(_shell, '--os-cloud megacloud module list')
        self.assertEqual('megacloud', _shell.cloud.name)
        self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
        self.assertEqual('cake', _shell.cloud.config['donut'])
        self.assertEqual('heart-o-gold', _shell.cloud.config['auth']['project_name'])
        self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
        self.assertEqual('occ-cloud', _shell.cloud.config['region_name'])
        self.assertEqual('occ-cloud', _shell.client_manager.region_name)
        self.assertEqual('mycert', _shell.cloud.config['cert'])
        self.assertEqual('mickey', _shell.cloud.config['key'])
        self.assertEqual(('mycert', 'mickey'), _shell.client_manager.cert)

    @mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
    @mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
    def test_shell_args_precedence(self, config_mock, vendor_mock):
        config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_2))
        vendor_mock.return_value = ('file.yaml', copy.deepcopy(PUBLIC_1))
        _shell = utils.make_shell()
        utils.fake_execute(_shell, '--os-cloud megacloud --os-region-name krikkit module list')
        self.assertEqual('megacloud', _shell.cloud.name)
        self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
        self.assertEqual('cake', _shell.cloud.config['donut'])
        self.assertEqual('heart-o-gold', _shell.cloud.config['auth']['project_name'])
        self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
        self.assertEqual('krikkit', _shell.cloud.config['region_name'])
        self.assertEqual('krikkit', _shell.client_manager.region_name)