import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class ConfLoadingLegacyTests(ConfLoadingTests):
    """Tests with inclusion of deprecated conf options.

    Not to be confused with ConfLoadingTests.test_deprecated, which tests
    external options that are deprecated in favor of Adapter options.
    """
    GROUP = 'adaptergroup'

    def setUp(self):
        super(ConfLoadingLegacyTests, self).setUp()
        self.conf_fx = self.useFixture(config.Config())
        loading.register_adapter_conf_options(self.conf_fx.conf, self.GROUP)

    def test_load_old_interface(self):
        self.conf_fx.config(service_type='type', service_name='name', interface='internal', region_name='region', endpoint_override='endpoint', version='2.0', group=self.GROUP)
        adap = loading.load_adapter_from_conf_options(self.conf_fx.conf, self.GROUP, session='session', auth='auth')
        self.assertEqual('type', adap.service_type)
        self.assertEqual('name', adap.service_name)
        self.assertEqual('internal', adap.interface)
        self.assertEqual('region', adap.region_name)
        self.assertEqual('endpoint', adap.endpoint_override)
        self.assertEqual('session', adap.session)
        self.assertEqual('auth', adap.auth)
        self.assertEqual('2.0', adap.version)
        self.assertIsNone(adap.min_version)
        self.assertIsNone(adap.max_version)

    def test_interface_conflict(self):
        self.conf_fx.config(service_type='type', service_name='name', interface='iface', valid_interfaces='internal,public', region_name='region', endpoint_override='endpoint', group=self.GROUP)
        self.assertRaises(TypeError, loading.load_adapter_from_conf_options, self.conf_fx.conf, self.GROUP, session='session', auth='auth')