from mistralclient.api.v2 import services
from mistralclient.tests.unit.v2 import base
class TestServicesV2(base.BaseClientV2Test):

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'services': [SERVICE]})
        service_list = self.services.list()
        self.assertEqual(1, len(service_list))
        srv = service_list[0]
        self.assertDictEqual(services.Service(self.services, SERVICE).to_dict(), srv.to_dict())