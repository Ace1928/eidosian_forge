from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupList(TestGroup):
    domain = identity_fakes.FakeDomain.create_one_domain()
    group = identity_fakes.FakeGroup.create_one_group()
    user = identity_fakes.FakeUser.create_one_user()
    columns = ('ID', 'Name')
    datalist = ((group.id, group.name),)

    def setUp(self):
        super(TestGroupList, self).setUp()
        self.groups_mock.get.return_value = self.group
        self.groups_mock.list.return_value = [self.group]
        self.domains_mock.get.return_value = self.domain
        self.users_mock.get.return_value = self.user
        self.cmd = group.ListGroup(self.app, None)

    def test_group_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'domain': None, 'user': None}
        self.groups_mock.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_group_list_domain(self):
        arglist = ['--domain', self.domain.id]
        verifylist = [('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'domain': self.domain.id, 'user': None}
        self.groups_mock.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_group_list_user(self):
        arglist = ['--user', self.user.name]
        verifylist = [('user', self.user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'domain': None, 'user': self.user.id}
        self.groups_mock.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_group_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'domain': None, 'user': None}
        self.groups_mock.list.assert_called_with(**kwargs)
        columns = self.columns + ('Domain ID', 'Description')
        datalist = ((self.group.id, self.group.name, self.group.domain_id, self.group.description),)
        self.assertEqual(columns, columns)
        self.assertEqual(datalist, tuple(data))