from openstackclient.identity.v2_0 import endpoint
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestEndpointDelete(TestEndpoint):

    def setUp(self):
        super(TestEndpointDelete, self).setUp()
        self.endpoints_mock.get.return_value = self.fake_endpoint
        self.endpoints_mock.delete.return_value = None
        self.cmd = endpoint.DeleteEndpoint(self.app, None)

    def test_endpoint_delete_no_options(self):
        arglist = [self.fake_endpoint.id]
        verifylist = [('endpoints', [self.fake_endpoint.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.endpoints_mock.delete.assert_called_with(self.fake_endpoint.id)
        self.assertIsNone(result)