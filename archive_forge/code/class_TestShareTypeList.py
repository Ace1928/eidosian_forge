from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeList(TestShareType):
    share_types = manila_fakes.FakeShareType.create_share_types()
    columns = utils.format_column_headers(COLUMNS)

    def setUp(self):
        super(TestShareTypeList, self).setUp()
        self.shares_mock.list.return_value = self.share_types
        self.cmd = osc_share_types.ListShareType(self.app, None)
        self.values = (oscutils.get_dict_properties(s._info, COLUMNS) for s in self.share_types)

    def test_share_type_list_no_options(self):
        arglist = []
        verifylist = [('all', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.list.assert_called_once_with(search_opts={}, show_all=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_type_list_all(self):
        arglist = ['--all']
        verifylist = [('all', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.list.assert_called_once_with(search_opts={}, show_all=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_type_list_extra_specs(self):
        arglist = ['--extra-specs', 'snapshot_support=true']
        verifylist = [('extra_specs', ['snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.list.assert_called_once_with(search_opts={'extra_specs': {'snapshot_support': 'True'}}, show_all=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_type_list_api_versions_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.42')
        arglist = ['--extra-specs', 'snapshot_support=true']
        verifylist = [('extra_specs', ['snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)