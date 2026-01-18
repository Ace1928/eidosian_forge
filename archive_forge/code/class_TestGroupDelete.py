from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupDelete(TestGroup):
    domain = identity_fakes.FakeDomain.create_one_domain()
    groups = identity_fakes.FakeGroup.create_groups(attrs={'domain_id': domain.id}, count=2)

    def setUp(self):
        super(TestGroupDelete, self).setUp()
        self.groups_mock.get = identity_fakes.FakeGroup.get_groups(self.groups)
        self.groups_mock.delete.return_value = None
        self.domains_mock.get.return_value = self.domain
        self.cmd = group.DeleteGroup(self.app, None)

    def test_group_delete(self):
        arglist = [self.groups[0].id]
        verifylist = [('groups', [self.groups[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_called_once_with(self.groups[0].id)
        self.groups_mock.delete.assert_called_once_with(self.groups[0].id)
        self.assertIsNone(result)

    def test_group_multi_delete(self):
        arglist = []
        verifylist = []
        for g in self.groups:
            arglist.append(g.id)
        verifylist = [('groups', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for g in self.groups:
            calls.append(call(g.id))
        self.groups_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_group_delete_with_domain(self):
        get_mock_result = [exceptions.CommandError, self.groups[0]]
        self.groups_mock.get = mock.Mock(side_effect=get_mock_result)
        arglist = ['--domain', self.domain.id, self.groups[0].id]
        verifylist = [('domain', self.groups[0].domain_id), ('groups', [self.groups[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_any_call(self.groups[0].id, domain_id=self.domain.id)
        self.groups_mock.delete.assert_called_once_with(self.groups[0].id)
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_delete_multi_groups_with_exception(self, find_mock):
        find_mock.side_effect = [self.groups[0], exceptions.CommandError]
        arglist = [self.groups[0].id, 'unexist_group']
        verifylist = [('groups', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 groups failed to delete.', str(e))
        find_mock.assert_any_call(self.groups_mock, self.groups[0].id)
        find_mock.assert_any_call(self.groups_mock, 'unexist_group')
        self.assertEqual(2, find_mock.call_count)
        self.groups_mock.delete.assert_called_once_with(self.groups[0].id)