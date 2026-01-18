import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class GetTransportTestCase(test_utils.BaseTestCase):
    scenarios = [('default', dict(url=None, transport_url=None, control_exchange=None, allowed=None, expect=dict(backend='rabbit', exchange=None, url='rabbit:', allowed=[]))), ('transport_url', dict(url=None, transport_url='testtransport:', control_exchange=None, allowed=None, expect=dict(backend='testtransport', exchange=None, url='testtransport:', allowed=[]))), ('url_param', dict(url='testtransport:', transport_url=None, control_exchange=None, allowed=None, expect=dict(backend='testtransport', exchange=None, url='testtransport:', allowed=[]))), ('control_exchange', dict(url=None, transport_url='testbackend:', control_exchange='testexchange', allowed=None, expect=dict(backend='testbackend', exchange='testexchange', url='testbackend:', allowed=[]))), ('allowed_remote_exmods', dict(url=None, transport_url='testbackend:', control_exchange=None, allowed=['foo', 'bar'], expect=dict(backend='testbackend', exchange=None, url='testbackend:', allowed=['foo', 'bar'])))]

    @mock.patch('oslo_messaging.transport.LOG')
    def test_get_transport(self, fake_logger):
        self.messaging_conf.reset()
        self.config(control_exchange=self.control_exchange)
        if self.transport_url:
            self.config(transport_url=self.transport_url)
        driver.DriverManager = mock.Mock()
        invoke_args = [self.conf, oslo_messaging.TransportURL.parse(self.conf, self.expect['url'])]
        invoke_kwds = dict(default_exchange=self.expect['exchange'], allowed_remote_exmods=self.expect['allowed'])
        drvr = _FakeDriver(self.conf)
        driver.DriverManager.return_value = _FakeManager(drvr)
        kwargs = dict(url=self.url)
        if self.allowed is not None:
            kwargs['allowed_remote_exmods'] = self.allowed
        transport_ = oslo_messaging.get_transport(self.conf, **kwargs)
        self.assertIsNotNone(transport_)
        self.assertIs(transport_.conf, self.conf)
        self.assertIs(transport_._driver, drvr)
        self.assertIsInstance(transport_, transport.RPCTransport)
        driver.DriverManager.assert_called_once_with('oslo.messaging.drivers', self.expect['backend'], invoke_on_load=True, invoke_args=invoke_args, invoke_kwds=invoke_kwds)