from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestEndpointSet(TestEndpoint):
    service = identity_fakes.FakeService.create_one_service()
    endpoint = identity_fakes.FakeEndpoint.create_one_endpoint(attrs={'service_id': service.id})

    def setUp(self):
        super(TestEndpointSet, self).setUp()
        self.endpoints_mock.get.return_value = self.endpoint
        self.endpoints_mock.update.return_value = self.endpoint
        self.services_mock.get.return_value = self.service
        self.cmd = endpoint.SetEndpoint(self.app, None)

    def test_endpoint_set_no_options(self):
        arglist = [self.endpoint.id]
        verifylist = [('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': None, 'interface': None, 'region': None, 'service': None, 'url': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_interface(self):
        arglist = ['--interface', 'public', self.endpoint.id]
        verifylist = [('interface', 'public'), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': None, 'interface': 'public', 'url': None, 'region': None, 'service': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_url(self):
        arglist = ['--url', 'http://localhost:5000', self.endpoint.id]
        verifylist = [('url', 'http://localhost:5000'), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': None, 'interface': None, 'url': 'http://localhost:5000', 'region': None, 'service': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_service(self):
        arglist = ['--service', self.service.id, self.endpoint.id]
        verifylist = [('service', self.service.id), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': None, 'interface': None, 'url': None, 'region': None, 'service': self.service.id}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_region(self):
        arglist = ['--region', 'e-rzzz', self.endpoint.id]
        verifylist = [('region', 'e-rzzz'), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': None, 'interface': None, 'url': None, 'region': 'e-rzzz', 'service': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_enable(self):
        arglist = ['--enable', self.endpoint.id]
        verifylist = [('enabled', True), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'interface': None, 'url': None, 'region': None, 'service': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_set_disable(self):
        arglist = ['--disable', self.endpoint.id]
        verifylist = [('disabled', True), ('endpoint', self.endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': False, 'interface': None, 'url': None, 'region': None, 'service': None}
        self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
        self.assertIsNone(result)