from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupSet(TestGroup):
    domain = identity_fakes.FakeDomain.create_one_domain()
    group = identity_fakes.FakeGroup.create_one_group(attrs={'domain_id': domain.id})

    def setUp(self):
        super(TestGroupSet, self).setUp()
        self.groups_mock.get.return_value = self.group
        self.domains_mock.get.return_value = self.domain
        self.groups_mock.update.return_value = None
        self.cmd = group.SetGroup(self.app, None)

    def test_group_set_nothing(self):
        arglist = [self.group.id]
        verifylist = [('group', self.group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.update.assert_called_once_with(self.group.id)
        self.assertIsNone(result)

    def test_group_set_name_and_description(self):
        arglist = ['--name', 'new_name', '--description', 'new_description', self.group.id]
        verifylist = [('name', 'new_name'), ('description', 'new_description'), ('group', self.group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'new_name', 'description': 'new_description'}
        self.groups_mock.update.assert_called_once_with(self.group.id, **kwargs)
        self.assertIsNone(result)

    def test_group_set_with_domain(self):
        get_mock_result = [exceptions.CommandError, self.group]
        self.groups_mock.get = mock.Mock(side_effect=get_mock_result)
        arglist = ['--domain', self.domain.id, self.group.id]
        verifylist = [('domain', self.domain.id), ('group', self.group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_any_call(self.group.id, domain_id=self.domain.id)
        self.groups_mock.update.assert_called_once_with(self.group.id)
        self.assertIsNone(result)