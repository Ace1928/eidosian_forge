import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc import utils
from manilaclient.osc.v2 import services as osc_services
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareServiceSet(TestShareService):

    def setUp(self):
        super(TestShareServiceSet, self).setUp()
        self.share_service = manila_fakes.FakeShareService.create_fake_service()
        self.cmd = osc_services.SetShareService(self.app, None)

    def test_share_service_set_enable(self):
        arglist = [self.share_service.host, self.share_service.binary, '--enable']
        verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.services_mock.enable.assert_called_with(self.share_service.host, self.share_service.binary)
        self.assertIsNone(result)

    def test_share_service_set_enable_exception(self):
        arglist = [self.share_service.host, self.share_service.binary, '--enable']
        verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.services_mock.enable.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_service_set_disable(self):
        arglist = [self.share_service.host, self.share_service.binary, '--disable']
        verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.services_mock.disable.assert_called_with(self.share_service.host, self.share_service.binary)
        self.assertIsNone(result)

    def test_service_set_disable_with_reason(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.83')
        reason = 'earthquake'
        arglist = ['--disable', '--disable-reason', reason, self.share_service.host, self.share_service.binary]
        verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('disable', True), ('disable_reason', reason)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.services_mock.disable.assert_called_with(self.share_service.host, self.share_service.binary, disable_reason=reason)
        self.assertIsNone(result)

    def test_share_service_set_disable_exception(self):
        arglist = [self.share_service.host, self.share_service.binary, '--disable']
        verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.services_mock.disable.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)