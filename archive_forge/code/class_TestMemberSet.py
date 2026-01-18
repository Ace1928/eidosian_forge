import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestMemberSet(TestMember):

    def setUp(self):
        super().setUp()
        self.cmd = member.SetMember(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_set(self, mock_attrs):
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'name': 'new_name', 'backup': True}
        arglist = [self._mem.pool_id, self._mem.id, '--name', 'new_name', '--enable-backup']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('name', 'new_name'), ('enable_backup', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'name': 'new_name', 'backup': True}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_set_wait(self, mock_attrs, mock_wait):
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'name': 'new_name'}
        arglist = [self._mem.pool_id, self._mem.id, '--name', 'new_name', '--wait']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('name', 'new_name'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'name': 'new_name'}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._mem.id, sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_set_tag(self, mock_attrs):
        self.api_mock.member_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'tags': ['bar']}
        arglist = [self._mem.pool_id, self._mem.id, '--tag', 'bar']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once()
        kwargs = self.api_mock.member_set.mock_calls[0][2]
        tags = kwargs['json']['member']['tags']
        self.assertEqual(2, len(tags))
        self.assertIn('foo', tags)
        self.assertIn('bar', tags)

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_set_tag_no_tag(self, mock_attrs):
        self.api_mock.member_show.return_value = {'tags': ['foo']}
        mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'tags': ['bar']}
        arglist = [self._mem.pool_id, self._mem.id, '--tag', 'bar', '--no-tag']
        verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'tags': ['bar']}})