import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc import utils
from manilaclient.osc.v2 import services as osc_services
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareServiceList(TestShareService):
    columns = ['id', 'binary', 'host', 'zone', 'status', 'state', 'updated_at']
    columns_with_reason = columns + ['disabled_reason']
    column_headers = utils.format_column_headers(columns)
    column_headers_with_reason = utils.format_column_headers(columns_with_reason)

    def setUp(self):
        super(TestShareServiceList, self).setUp()
        self.services_list = manila_fakes.FakeShareService.create_fake_services({'disabled_reason': ''})
        self.services_mock.list.return_value = self.services_list
        self.values = (oscutils.get_dict_properties(i._info, self.columns) for i in self.services_list)
        self.values_with_reason = (oscutils.get_dict_properties(i._info, self.columns_with_reason) for i in self.services_list)
        self.cmd = osc_services.ListShareService(self.app, None)

    @ddt.data('2.82', '2.83')
    def test_share_service_list(self, version):
        self.app.client_manager.share.api_version = api_versions.APIVersion(version)
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.services_mock.list.assert_called_with(search_opts={'host': None, 'binary': None, 'status': None, 'state': None, 'zone': None})
        if api_versions.APIVersion(version) >= api_versions.APIVersion('2.83'):
            self.assertEqual(self.column_headers_with_reason, columns)
            self.assertEqual(list(self.values_with_reason), list(data))
        else:
            self.assertEqual(self.column_headers, columns)
            self.assertEqual(list(self.values), list(data))

    @ddt.data('2.82', '2.83')
    def test_share_service_list_host_status(self, version):
        self.app.client_manager.share.api_version = api_versions.APIVersion(version)
        arglist = ['--host', self.services_list[0].host, '--status', self.services_list[1].status]
        verifylist = [('host', self.services_list[0].host), ('status', self.services_list[1].status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.services_mock.list.assert_called_with(search_opts={'host': self.services_list[0].host, 'binary': None, 'status': self.services_list[1].status, 'state': None, 'zone': None})
        if api_versions.APIVersion(version) >= api_versions.APIVersion('2.83'):
            self.assertEqual(self.column_headers_with_reason, columns)
            self.assertEqual(list(self.values_with_reason), list(data))
        else:
            self.assertEqual(self.column_headers, columns)
            self.assertEqual(list(self.values), list(data))

    @ddt.data('2.82', '2.83')
    def test_share_service_list_binary_state_zone(self, version):
        self.app.client_manager.share.api_version = api_versions.APIVersion(version)
        arglist = ['--binary', self.services_list[0].binary, '--state', self.services_list[1].state, '--zone', self.services_list[1].zone]
        verifylist = [('binary', self.services_list[0].binary), ('state', self.services_list[1].state), ('zone', self.services_list[1].zone)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.services_mock.list.assert_called_with(search_opts={'host': None, 'binary': self.services_list[0].binary, 'status': None, 'state': self.services_list[1].state, 'zone': self.services_list[1].zone})
        if api_versions.APIVersion(version) >= api_versions.APIVersion('2.83'):
            self.assertEqual(self.column_headers_with_reason, columns)
            self.assertEqual(list(self.values_with_reason), list(data))
        else:
            self.assertEqual(self.column_headers, columns)
            self.assertEqual(list(self.values), list(data))