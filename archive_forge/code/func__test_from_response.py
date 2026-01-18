from novaclient import exceptions
from novaclient.tests.unit import utils as test_utils
def _test_from_response(self, body, expected_message):
    data = {'status_code': 404, 'headers': {'content-type': 'application/json', 'x-openstack-request-id': 'req-d9df03b0-4150-4b53-8157-7560ccf39f75'}}
    response = test_utils.TestResponse(data)
    fake_url = 'http://localhost:8774/v2.1/fake/flavors/test'
    error = exceptions.from_response(response, body, fake_url, 'GET')
    self.assertIsInstance(error, exceptions.NotFound)
    self.assertEqual(expected_message, error.message)