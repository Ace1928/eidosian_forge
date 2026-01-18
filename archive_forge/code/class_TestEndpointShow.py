from openstackclient.identity.v2_0 import endpoint
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestEndpointShow(TestEndpoint):

    def setUp(self):
        super(TestEndpointShow, self).setUp()
        self.endpoints_mock.list.return_value = [self.fake_endpoint]
        self.services_mock.get.return_value = self.fake_service
        self.cmd = endpoint.ShowEndpoint(self.app, None)

    def test_endpoint_show(self):
        arglist = [self.fake_endpoint.id]
        verifylist = [('endpoint_or_service', self.fake_endpoint.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.endpoints_mock.list.assert_called_with()
        self.services_mock.get.assert_called_with(self.fake_endpoint.service_id)
        collist = ('adminurl', 'id', 'internalurl', 'publicurl', 'region', 'service_id', 'service_name', 'service_type')
        self.assertEqual(collist, columns)
        datalist = (self.fake_endpoint.adminurl, self.fake_endpoint.id, self.fake_endpoint.internalurl, self.fake_endpoint.publicurl, self.fake_endpoint.region, self.fake_endpoint.service_id, self.fake_endpoint.service_name, self.fake_endpoint.service_type)
        self.assertEqual(datalist, data)