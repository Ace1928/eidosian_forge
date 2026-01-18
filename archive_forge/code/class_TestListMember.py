import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListMember(TestMember):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.MEMBER_ATTRS[k] for k in self.columns)),)
        self.cmd = member.ListMember(self.app, None)

    def test_member_list_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_list(self, mock_attrs):
        mock_attrs.return_value = {'pool_id': 'pool_id', 'project_id': self._mem.project_id}
        arglist = ['pool_id']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.member_list.assert_called_once_with(pool_id='pool_id', project_id=self._mem.project_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_member_list_with_tags(self):
        arglist = [self._mem.pool_id, '--tags', 'foo,bar']
        verifylist = [('pool', self._mem.pool_id), ('tags', ['foo', 'bar'])]
        expected_attrs = {'pool_id': self._mem.pool_id, 'tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.member_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_member_list_with_any_tags(self):
        arglist = [self._mem.pool_id, '--any-tags', 'foo,bar']
        verifylist = [('pool', self._mem.pool_id), ('any_tags', ['foo', 'bar'])]
        expected_attrs = {'pool_id': self._mem.pool_id, 'tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.member_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_member_list_with_not_tags(self):
        arglist = [self._mem.pool_id, '--not-tags', 'foo,bar']
        verifylist = [('pool', self._mem.pool_id), ('not_tags', ['foo', 'bar'])]
        expected_attrs = {'pool_id': self._mem.pool_id, 'not-tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.member_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_member_list_with_not_any_tags(self):
        arglist = [self._mem.pool_id, '--not-any-tags', 'foo,bar']
        verifylist = [('pool', self._mem.pool_id), ('not_any_tags', ['foo', 'bar'])]
        expected_attrs = {'pool_id': self._mem.pool_id, 'not-tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.member_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))