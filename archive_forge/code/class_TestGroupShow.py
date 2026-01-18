from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupShow(TestGroup):
    domain = identity_fakes.FakeDomain.create_one_domain()
    columns = ('description', 'domain_id', 'id', 'name')

    def setUp(self):
        super(TestGroupShow, self).setUp()
        self.group = identity_fakes.FakeGroup.create_one_group(attrs={'domain_id': self.domain.id})
        self.data = (self.group.description, self.group.domain_id, self.group.id, self.group.name)
        self.groups_mock.get.return_value = self.group
        self.domains_mock.get.return_value = self.domain
        self.cmd = group.ShowGroup(self.app, None)

    def test_group_show(self):
        arglist = [self.group.id]
        verifylist = [('group', self.group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_called_once_with(self.group.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_group_show_with_domain(self):
        get_mock_result = [exceptions.CommandError, self.group]
        self.groups_mock.get = mock.Mock(side_effect=get_mock_result)
        arglist = ['--domain', self.domain.id, self.group.id]
        verifylist = [('domain', self.domain.id), ('group', self.group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_any_call(self.group.id, domain_id=self.domain.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)