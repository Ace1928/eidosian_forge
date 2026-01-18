from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupCreate(TestGroup):
    domain = identity_fakes.FakeDomain.create_one_domain()
    columns = ('description', 'domain_id', 'id', 'name')

    def setUp(self):
        super(TestGroupCreate, self).setUp()
        self.group = identity_fakes.FakeGroup.create_one_group(attrs={'domain_id': self.domain.id})
        self.data = (self.group.description, self.group.domain_id, self.group.id, self.group.name)
        self.groups_mock.create.return_value = self.group
        self.groups_mock.get.return_value = self.group
        self.domains_mock.get.return_value = self.domain
        self.cmd = group.CreateGroup(self.app, None)

    def test_group_create(self):
        arglist = [self.group.name]
        verifylist = [('name', self.group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_once_with(name=self.group.name, domain=None, description=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_group_create_with_options(self):
        arglist = ['--domain', self.domain.name, '--description', self.group.description, self.group.name]
        verifylist = [('domain', self.domain.name), ('description', self.group.description), ('name', self.group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_once_with(name=self.group.name, domain=self.domain.id, description=self.group.description)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_group_create_or_show(self):
        self.groups_mock.create.side_effect = ks_exc.Conflict()
        arglist = ['--or-show', self.group.name]
        verifylist = [('or_show', True), ('name', self.group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_called_once_with(self.group.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)