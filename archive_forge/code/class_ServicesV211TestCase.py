from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import services
class ServicesV211TestCase(ServicesTest):
    api_version = '2.11'

    def _update_body(self, host, binary, disabled_reason=None, force_down=None):
        body = {'host': host, 'binary': binary}
        if disabled_reason is not None:
            body['disabled_reason'] = disabled_reason
        if force_down is not None:
            body['forced_down'] = force_down
        return body

    def test_services_force_down(self):
        service = self.cs.services.force_down('compute1', 'nova-compute', False)
        self.assert_request_id(service, fakes.FAKE_REQUEST_ID_LIST)
        values = self._update_body('compute1', 'nova-compute', force_down=False)
        self.cs.assert_called('PUT', '/os-services/force-down', values)
        self.assertIsInstance(service, self._get_service_type())
        self.assertFalse(service.forced_down)