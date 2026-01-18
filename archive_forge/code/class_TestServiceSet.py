from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import service
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestServiceSet(TestService):
    service = identity_fakes.FakeService.create_one_service()

    def setUp(self):
        super(TestServiceSet, self).setUp()
        self.services_mock.get.side_effect = identity_exc.NotFound(None)
        self.services_mock.find.return_value = self.service
        self.services_mock.update.return_value = self.service
        self.cmd = service.SetService(self.app, None)

    def test_service_set_no_options(self):
        arglist = [self.service.name]
        verifylist = [('type', None), ('name', None), ('description', None), ('enable', False), ('disable', False), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)

    def test_service_set_type(self):
        arglist = ['--type', self.service.type, self.service.name]
        verifylist = [('type', self.service.type), ('name', None), ('description', None), ('enable', False), ('disable', False), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'type': self.service.type}
        self.services_mock.update.assert_called_with(self.service.id, **kwargs)
        self.assertIsNone(result)

    def test_service_set_name(self):
        arglist = ['--name', self.service.name, self.service.name]
        verifylist = [('type', None), ('name', self.service.name), ('description', None), ('enable', False), ('disable', False), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': self.service.name}
        self.services_mock.update.assert_called_with(self.service.id, **kwargs)
        self.assertIsNone(result)

    def test_service_set_description(self):
        arglist = ['--description', self.service.description, self.service.name]
        verifylist = [('type', None), ('name', None), ('description', self.service.description), ('enable', False), ('disable', False), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.service.description}
        self.services_mock.update.assert_called_with(self.service.id, **kwargs)
        self.assertIsNone(result)

    def test_service_set_enable(self):
        arglist = ['--enable', self.service.name]
        verifylist = [('type', None), ('name', None), ('description', None), ('enable', True), ('disable', False), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True}
        self.services_mock.update.assert_called_with(self.service.id, **kwargs)
        self.assertIsNone(result)

    def test_service_set_disable(self):
        arglist = ['--disable', self.service.name]
        verifylist = [('type', None), ('name', None), ('description', None), ('enable', False), ('disable', True), ('service', self.service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': False}
        self.services_mock.update.assert_called_with(self.service.id, **kwargs)
        self.assertIsNone(result)