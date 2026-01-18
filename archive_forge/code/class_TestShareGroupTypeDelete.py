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
class TestShareGroupTypeDelete(TestShareGroupType):

    def setUp(self):
        super(TestShareGroupTypeDelete, self).setUp()
        self.share_group_types = manila_fakes.FakeShareGroupType.create_share_group_types(count=2)
        self.sgt_mock.delete.return_value = None
        self.sgt_mock.get = manila_fakes.FakeShareGroupType.get_share_group_types(self.share_group_types)
        self.cmd = osc_share_group_types.DeleteShareGroupType(self.app, None)

    def test_share_group_type_delete_one(self):
        arglist = [self.share_group_types[0].name]
        verifylist = [('share_group_types', [self.share_group_types[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.sgt_mock.delete.assert_called_with(self.share_group_types[0])
        self.assertIsNone(result)

    def test_share_group_type_delete_multiple(self):
        arglist = []
        for t in self.share_group_types:
            arglist.append(t.name)
        verifylist = [('share_group_types', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for t in self.share_group_types:
            calls.append(mock.call(t))
        self.sgt_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_share_group_type_with_exception(self):
        arglist = ['non_existing_type']
        verifylist = [('share_group_types', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.sgt_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_delete_share_group_type(self):
        arglist = [self.share_group_types[0].name]
        verifylist = [('share_group_types', [self.share_group_types[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.sgt_mock.delete.assert_called_with(self.share_group_types[0])
        self.assertIsNone(result)