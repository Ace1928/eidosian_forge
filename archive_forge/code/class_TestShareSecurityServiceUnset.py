import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareSecurityServiceUnset(TestShareSecurityService):

    def setUp(self):
        super(TestShareSecurityServiceUnset, self).setUp()
        self.security_service = manila_fakes.FakeShareSecurityService.create_fake_security_service(methods={'update': None})
        self.security_services_mock.get.return_value = self.security_service
        self.cmd = osc_security_services.UnsetShareSecurityService(self.app, None)

    def test_share_security_service_unset_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_security_service_unset(self):
        arglist = [self.security_service.id, '--dns-ip', '--ou', '--server', '--domain', '--user', '--password', '--name', '--description', '--default-ad-site']
        verifylist = [('security_service', self.security_service.id), ('dns_ip', True), ('ou', True), ('server', True), ('domain', True), ('user', True), ('password', True), ('name', True), ('description', True), ('default_ad_site', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.security_service.update.assert_called_with(dns_ip='', server='', domain='', user='', password='', name='', description='', ou='', default_ad_site='')
        self.assertIsNone(result)

    def test_share_security_service_unset_exception(self):
        arglist = [self.security_service.id, '--name']
        verifylist = [('security_service', self.security_service.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.security_service.update.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @ddt.data('2.43', '2.75')
    def test_share_security_service_unset_api_version_exception(self, version):
        self.app.client_manager.share.api_version = api_versions.APIVersion(version)
        arglist = [self.security_service.id]
        verifylist = [('security_service', self.security_service.id)]
        if api_versions.APIVersion(version) <= api_versions.APIVersion('2.43'):
            arglist.extend(['--ou'])
            verifylist.append(('ou', True))
        if api_versions.APIVersion(version) <= api_versions.APIVersion('2.75'):
            (arglist.extend(['--default-ad-site']),)
            verifylist.append(('default_ad_site', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)