import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSecurityServiceShow(TestShareSecurityService):

    def setUp(self):
        super(TestShareSecurityServiceShow, self).setUp()
        self.security_service = manila_fakes.FakeShareSecurityService.create_fake_security_service()
        self.security_services_mock.get.return_value = self.security_service
        self.cmd = osc_security_services.ShowShareSecurityService(self.app, None)
        self.data = self.security_service._info.values()
        self.columns = self.security_service._info.keys()

    def test_share_security_service_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_security_service_show(self):
        arglist = [self.security_service.id]
        verifylist = [('security_service', self.security_service.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.security_services_mock.get.assert_called_with(self.security_service.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)