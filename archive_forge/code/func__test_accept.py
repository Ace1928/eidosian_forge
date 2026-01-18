from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def _test_accept(self, client, expected_url):
    transfer_id = '5678'
    auth_key = '12345'
    vol = client.transfers.accept(transfer_id, auth_key)
    client.assert_called('POST', '/%s/%s/accept' % (expected_url, transfer_id))
    self._assert_request_id(vol)