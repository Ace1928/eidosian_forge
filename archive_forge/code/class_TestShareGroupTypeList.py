from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupTypeList(TestShareGroupType):

    def setUp(self):
        super(TestShareGroupTypeList, self).setUp()
        self.share_group_types = manila_fakes.FakeShareGroupType.create_share_group_types()
        self.sgt_mock.list.return_value = self.share_group_types
        self.cmd = osc_share_group_types.ListShareGroupType(self.app, None)
        self.values = (oscutils.get_dict_properties(s._info, COLUMNS) for s in self.share_group_types)

    def test_share_group_type_list_no_options(self):
        arglist = []
        verifylist = [('all', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sgt_mock.list.assert_called_once_with(search_opts={}, show_all=False)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_type_list_all(self):
        arglist = ['--all']
        verifylist = [('all', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sgt_mock.list.assert_called_once_with(search_opts={}, show_all=True)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_type_list_group_specs(self):
        arglist = ['--group-specs', 'consistent_snapshot_support=true']
        verifylist = [('group_specs', ['consistent_snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sgt_mock.list.assert_called_once_with(search_opts={'group_specs': {'consistent_snapshot_support': 'True'}}, show_all=False)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))