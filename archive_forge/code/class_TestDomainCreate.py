from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestDomainCreate(TestDomain):
    columns = ('description', 'enabled', 'id', 'name', 'tags')

    def setUp(self):
        super(TestDomainCreate, self).setUp()
        self.domain = identity_fakes.FakeDomain.create_one_domain()
        self.domains_mock.create.return_value = self.domain
        self.datalist = (self.domain.description, True, self.domain.id, self.domain.name, self.domain.tags)
        self.cmd = domain.CreateDomain(self.app, None)

    def test_domain_create_no_options(self):
        arglist = [self.domain.name]
        verifylist = [('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': None, 'options': {}, 'enabled': True}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_domain_create_description(self):
        arglist = ['--description', 'new desc', self.domain.name]
        verifylist = [('description', 'new desc'), ('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': 'new desc', 'options': {}, 'enabled': True}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_domain_create_enable(self):
        arglist = ['--enable', self.domain.name]
        verifylist = [('enable', True), ('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': None, 'options': {}, 'enabled': True}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_domain_create_disable(self):
        arglist = ['--disable', self.domain.name]
        verifylist = [('disable', True), ('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': None, 'options': {}, 'enabled': False}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_domain_create_with_immutable(self):
        arglist = ['--immutable', self.domain.name]
        verifylist = [('immutable', True), ('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': None, 'options': {'immutable': True}, 'enabled': True}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_domain_create_with_no_immutable(self):
        arglist = ['--no-immutable', self.domain.name]
        verifylist = [('no_immutable', True), ('name', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.domain.name, 'description': None, 'options': {'immutable': False}, 'enabled': True}
        self.domains_mock.create.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)