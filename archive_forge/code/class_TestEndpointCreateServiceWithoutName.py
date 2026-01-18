from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestEndpointCreateServiceWithoutName(TestEndpointCreate):
    service = identity_fakes.FakeService.create_one_service(attrs={'service_name': ''})

    def setUp(self):
        super(TestEndpointCreate, self).setUp()
        self.endpoint = identity_fakes.FakeEndpoint.create_one_endpoint(attrs={'service_id': self.service.id})
        self.endpoints_mock.create.return_value = self.endpoint
        self.services_mock.get.return_value = self.service
        self.cmd = endpoint.CreateEndpoint(self.app, None)