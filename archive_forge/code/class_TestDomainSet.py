from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestDomainSet(TestDomain):
    domain = identity_fakes.FakeDomain.create_one_domain()

    def setUp(self):
        super(TestDomainSet, self).setUp()
        self.domains_mock.get.return_value = self.domain
        self.domains_mock.update.return_value = self.domain
        self.cmd = domain.SetDomain(self.app, None)

    def test_domain_set_no_options(self):
        arglist = [self.domain.name]
        verifylist = [('domain', self.domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_name(self):
        arglist = ['--name', 'qwerty', self.domain.id]
        verifylist = [('name', 'qwerty'), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'qwerty'}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_description(self):
        arglist = ['--description', 'new desc', self.domain.id]
        verifylist = [('description', 'new desc'), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': 'new desc'}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_enable(self):
        arglist = ['--enable', self.domain.id]
        verifylist = [('enable', True), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_disable(self):
        arglist = ['--disable', self.domain.id]
        verifylist = [('disable', True), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': False}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_immutable_option(self):
        arglist = ['--immutable', self.domain.id]
        verifylist = [('immutable', True), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'options': {'immutable': True}}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)

    def test_domain_set_no_immutable_option(self):
        arglist = ['--no-immutable', self.domain.id]
        verifylist = [('no_immutable', True), ('domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'options': {'immutable': False}}
        self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
        self.assertIsNone(result)