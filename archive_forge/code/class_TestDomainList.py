from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestDomainList(TestDomain):
    domain = identity_fakes.FakeDomain.create_one_domain()

    def setUp(self):
        super(TestDomainList, self).setUp()
        self.domains_mock.list.return_value = [self.domain]
        self.cmd = domain.ListDomain(self.app, None)

    def test_domain_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.domains_mock.list.assert_called_with()
        collist = ('ID', 'Name', 'Enabled', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((self.domain.id, self.domain.name, True, self.domain.description),)
        self.assertEqual(datalist, tuple(data))

    def test_domain_list_with_option_name(self):
        arglist = ['--name', self.domain.name]
        verifylist = [('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name}
        self.domains_mock.list.assert_called_with(**kwargs)
        collist = ('ID', 'Name', 'Enabled', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((self.domain.id, self.domain.name, True, self.domain.description),)
        self.assertEqual(datalist, tuple(data))

    def test_domain_list_with_option_enabled(self):
        arglist = ['--enabled']
        verifylist = [('enabled', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True}
        self.domains_mock.list.assert_called_with(**kwargs)
        collist = ('ID', 'Name', 'Enabled', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((self.domain.id, self.domain.name, True, self.domain.description),)
        self.assertEqual(datalist, tuple(data))